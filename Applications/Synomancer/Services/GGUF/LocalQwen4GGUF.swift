import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM

/// Owned by LocalChatService's serial generation gate, alongside the Swift models.
actor LocalQwen4GGUF {
    let directory: URL
    private var mtp: Qwen4GGUFMTP?
    private var model: Qwen4GGUFModel?
    private var vision: Qwen4GGUFVision.VisionModel?
    private var tokenizer: (any MLXLMCommon.Tokenizer)?
    private let ticket = WiredMemoryTicket(size: 105 * 1024 * 1024 * 1024, policy: WiredMaxPolicy())

    init(directory: URL) { self.directory = directory }

    func generate(messages: [LocalChatMessage], temperature: Float, maxTokens: Int,
                  topP: Float, stop: [String], useMTP: Bool, prefillStepSize: Int? = nil,
                  onUpdate: @escaping @Sendable (String, Int) -> Void) async throws -> LocalChatResult {
        guard ProcessInfo.processInfo.physicalMemory >= 120 * 1024 * 1024 * 1024 else {
            throw GGUFFile.Failure(message: "This resident IQ3_S model requires 128 GB unified memory")
        }
        guard maxTokens > 0, (0...2).contains(temperature), topP > 0, topP <= 1 else {
            throw GGUFFile.Failure(message: "Invalid generation parameters")
        }
        _ = await ticket.start()
        do {
            try Task.checkCancellation()
            if tokenizer == nil { tokenizer = try await LocalChatTokenizerLoader().load(from: directory) }
            let tokenizer = tokenizer!
            let descriptor = try Qwen4GGUFDescriptor.read(directory)
            var images: [UserInput.Image] = []
            let history: [MLXLMCommon.Message] = try messages.map { item in
                if let data = item.image {
                    guard let image = CIImage(data: data, options: [.applyOrientationProperty: true]) else { throw LocalChatService.ChatError.invalidImage }
                    images.append(.ciImage(image))
                    return ["role": item.role, "content": [["type": "image"], ["type": "text", "text": item.content]]]
                }
                return ["role": item.role, "content": item.content]
            }
            var userInput = UserInput(messages: history, images: images)
            userInput.additionalContext = ["enable_thinking": false]
            // Bound global vision attention independently of text prefill batching.
            userInput.processing.resize = CGSize(width: 448, height: 448)
            let config = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self,
                from: Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json")))
            let prepared = try await Qwen3VLProcessor(config, tokenizer: tokenizer).prepare(input: userInput)
            let tokens = prepared.text.tokens.asArray(Int.self)
            guard !tokens.isEmpty, tokens.count + maxTokens <= 32768 else {
                throw GGUFFile.Failure(message: "Native Qwen4 context limit is 32768 tokens")
            }
            try Task.checkCancellation()
            if model == nil {
                MLX.Memory.cacheLimit = 1 << 30
                model = try Qwen4GGUFModel(firstShard: directory.appendingPathComponent(descriptor.firstShard))
            }
            let model = model!
            model.reset()
            defer { model.reset(); mtp?.reset() }
            var fallback: String? = !useMTP ? "MTP is disabled" : temperature != 0 ? "Qwen MTP requires temperature 0" : nil
            let useDraft = useMTP && temperature == 0 && descriptor.mtp != nil
            if useMTP && temperature == 0 && descriptor.mtp == nil { fallback = "No complete native MTP sidecar configured" }
            if useDraft, mtp == nil { mtp = try Qwen4GGUFMTP(sidecar: directory.appendingPathComponent(descriptor.mtp!)) }
            mtp?.reset()
            if let image = prepared.image {
                guard let projector = descriptor.projector else { throw LocalChatService.ChatError.imagesUnsupported }
                if vision == nil { vision = try Qwen4GGUFVision.VisionModel.load(directory: directory, projector: directory.appendingPathComponent(projector)) }
                guard let frames = image.frames else { throw LocalChatService.ChatError.invalidImage }
                let (features, _) = vision!(image.pixels, gridTHW: frames)
                eval(features)
                try prepareVision(model, tokens: tokens, features: features, grids: frames)
            }
            let started = Date()
            var logits = MLXArray(0)
            let step = min(1024, max(1, prefillStepSize ?? (tokens.count >= 1024 ? 1024 : tokens.count >= 128 ? 512 : 16)))
            let bulk = tokens.count >= 128
            if useDraft { mtp!.warmupRounds = bulk ? 8 : 2 }
            // Match the generation API's final-token boundary for bulk prompts.
            // The draft starts from the target's final hidden row, which already
            // represents the full prompt; no second full-prompt pass is needed.
            for range in Qwen4GGUFModel.prefillRanges(count: tokens.count, step: step, separateLast: bulk) {
                try Task.checkCancellation()
                let start = range.lowerBound, end = range.upperBound
                logits = model.forward(Array(tokens[start..<end]), prefill: tokens.count >= 128, lastLogitsOnly: true).reshaped([-1])
                if useDraft && (!bulk || end == tokens.count) {
                    if bulk {
                        mtp!.prefill(shiftedTokens: [argMax(logits).item(Int.self)],
                            hidden: model.lastHidden[0..., (model.lastHidden.dim(1)-1)..., 0...], fast: false)
                    } else {
                        var shifted = Array(tokens[(start+1)..<min(end+1, tokens.count)])
                        if end == tokens.count { shifted.append(argMax(logits).item(Int.self)) }
                        mtp!.prefill(shiftedTokens: shifted, hidden: model.lastHidden, fast: false)
                    }
                }
            }
            eval(logits)
            let prefilled = Date()
            let sampler = GenerateParameters(temperature: temperature, topP: topP).sampler()
            var generated: [Int] = [], text = "", reason = "length"
            let eos = Set([tokenizer.eosTokenId, tokenizer.convertTokenToId("<|im_end|>"), tokenizer.convertTokenToId("<|endoftext|>")].compactMap { $0 })
            var pending = useDraft ? argMax(logits).item(Int.self) : 0
            generation: while generated.count < maxTokens {
                try Task.checkCancellation()
                let emitted: [Int]
                if useDraft {
                    let result = try mtp!.verify(root: pending, target: model, maxCount: maxTokens - generated.count)
                    emitted = result.tokens
                    if let next = result.next { pending = next }
                } else {
                    emitted = [sampler.sample(logits: logits.reshaped([1, -1])).item(Int.self)]
                }
                for token in emitted {
                    generated.append(token)
                    text = tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
                    onUpdate(LocalChatService.visibleText(text, stop: stop), generated.count)
                    if eos.contains(token) || stop.contains(where: { !$0.isEmpty && text.contains($0) }) {
                        reason = "stop"
                        break generation
                    }
                    if generated.count == maxTokens { break generation }
                }
                if !useDraft { logits = model.step(emitted[0]) }
            }
            try Task.checkCancellation()
            if let end = stop.filter({ !$0.isEmpty }).compactMap({ text.range(of: $0)?.lowerBound }).min() { text = String(text[..<end]) }
            let seconds = Date().timeIntervalSince(prefilled)
            onUpdate(text, generated.count)
            _ = await ticket.end()
            return LocalChatResult(text: text, tokensPerSecond: Double(max(0, generated.count-1)) / max(seconds, 1e-9),
                promptTokens: tokens.count, completionTokens: generated.count, finishReason: reason,
                proposedDraftTokens: useDraft ? mtp?.proposed : nil, acceptedDraftTokens: useDraft ? mtp?.accepted : nil,
                mtpFallbackReason: fallback,
                promptSeconds: prefilled.timeIntervalSince(started), generationSeconds: seconds)
        } catch {
            _ = await ticket.end()
            throw error
        }
    }

    private func prepareVision(_ model: Qwen4GGUFModel, tokens: [Int], features: MLXArray, grids: [THW]) throws {
        var index = 0, imageIndex = 0, featureIndex = 0, position = 0
        while index < tokens.count {
            if tokens[index] == 248056 {
                guard imageIndex < grids.count else { throw LocalChatService.ChatError.invalidImage }
                let grid = grids[imageIndex], h = grid.h / 2, w = grid.w / 2
                let count = grid.t * h * w
                guard count > 0, index + count <= tokens.count, featureIndex + count <= features.dim(0),
                      tokens[index..<(index+count)].allSatisfy({ $0 == 248056 }) else { throw LocalChatService.ChatError.invalidImage }
                for t in 0..<grid.t { for y in 0..<h { for x in 0..<w {
                    model.positionIDs.append([position + t, position + y, position + x])
                    model.imageEmbeddings[index] = features[featureIndex].reshaped([1, 1, 2560])
                    featureIndex += 1; index += 1
                } } }
                position += max(grid.t, max(h, w)); imageIndex += 1
            } else {
                model.positionIDs.append([position, position, position])
                position += 1; index += 1
            }
        }
        guard imageIndex == grids.count, featureIndex == features.dim(0) else { throw LocalChatService.ChatError.invalidImage }
    }
}
