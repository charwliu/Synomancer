import Foundation
@preconcurrency import CoreImage
import MLXLLM
import MLXVLM
import MLXLMCommon

struct LocalChatMessage: Sendable {
    let role: String
    let content: String
    var image: Data? = nil
}

struct LocalChatResult: Sendable {
    let text: String
    let tokensPerSecond: Double
    var promptTokens: Int = 0
    var completionTokens: Int = 0
    var finishReason: String = "stop"
    var proposedDraftTokens: Int? = nil
    var acceptedDraftTokens: Int? = nil
    var mtpFallbackReason: String? = nil
    var promptSeconds: Double = 0
    var generationSeconds: Double = 0
}

actor LocalChatService {
    static let shared = LocalChatService()
    // All model access is actor-owned and protected by the generating gate across
    // suspension points. The upstream worker is joined before another request runs.
    private var context: ModelContext?
    private var drafter: Qwen35MTPDraftModel?
    private var loadedDirectory: URL?
    private var loadedCache: SpeechCacheAccess.Configuration?
    private var generating = false

    func models(cache: SpeechCacheAccess.Configuration) async throws -> [LocalChatModel] {
        let root = try cache.resolve()
        let accessing = root.startAccessingSecurityScopedResource()
        defer { if accessing { root.stopAccessingSecurityScopedResource() } }
        _ = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        let candidates = try LocalChatModelCache.scan(in: root, allSnapshots: true)
        var llmTypes: Set<String> = [], vlmTypes: Set<String> = []
        for type in Set(candidates.map(\.modelType)) {
            if await LLMTypeRegistry.shared.contains(type) { llmTypes.insert(type) }
            if await VLMTypeRegistry.shared.contains(type) { vlmTypes.insert(type) }
        }
        return try LocalChatModelCache.scan(in: root, llmTypes: llmTypes, vlmTypes: vlmTypes)
    }

    func generate(modelID: String, messages: [LocalChatMessage], cache: SpeechCacheAccess.Configuration,
                  temperature: Float = 0.6, maxTokens: Int = 2048, topP: Float = 1, stop: [String] = [], useMTP: Bool = true, frameReasoning: Bool = false,
                  onUpdate: @escaping @Sendable (String, Int) -> Void) async throws -> LocalChatResult {
        while generating {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(20))
        }
        try Task.checkCancellation()
        generating = true
        defer { generating = false }
        let root = try cache.resolve()
        let accessing = root.startAccessingSecurityScopedResource()
        defer { if accessing { root.stopAccessingSecurityScopedResource() } }
        guard let model = try await models(cache: cache).first(where: { $0.id == modelID }) else {
            throw ChatError.modelUnavailable(modelID)
        }
        guard model.weightBytes < ProcessInfo.processInfo.physicalMemory * 4 / 5 else {
            throw ChatError.modelTooLarge(modelID)
        }
        let configData = try Data(contentsOf: model.directory.appendingPathComponent("config.json"))
        if let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            try LocalQwenMTP.validateGating(config)
        }
        if messages.contains(where: { $0.image != nil }) && !model.supportsImages { throw ChatError.imagesUnsupported }
        if context == nil || loadedDirectory != model.directory || loadedCache != cache {
            let tokenizerLoader = try await LocalChatTokenizerLoader.validated(for: model)
            try Task.checkCancellation()
            context = nil
            drafter = nil
            if model.isVisionModel {
                let factory = model.modelType == "qwen3_5_moe" ? LocalQwenMoEModel.factory : VLMModelFactory.shared
                context = try await factory.load(from: model.directory, using: tokenizerLoader)
            } else {
                context = try await LLMModelFactory.shared.load(from: model.directory, using: tokenizerLoader)
            }
            loadedDirectory = model.directory
            loadedCache = cache
        }
        try Task.checkCancellation()
        guard let context else { throw ChatError.modelUnavailable(modelID) }
        if useMTP && temperature == 0 && !model.isVisionModel && drafter == nil,
           let checkpoint = try LocalQwenMTP.checkpoint(in: model.directory) {
            guard model.weightBytes + checkpoint.bytes < ProcessInfo.processInfo.physicalMemory * 4 / 5 else {
                throw ChatError.modelTooLarge(modelID)
            }
            drafter = try LocalQwenMTP.load(checkpoint)
        }
        do {
            var images: [UserInput.Image] = []
            let history: [MLXLMCommon.Message] = try messages.map { message in
                if model.isVisionModel {
                    var content: [[String: String]] = [["type": "text", "text": message.content]]
                    if let data = message.image {
                        guard let image = CIImage(data: data) else { throw ChatError.invalidImage }
                        images.append(.ciImage(image))
                        content.append(["type": "image"])
                    }
                    return ["role": message.role, "content": content]
                }
                return ["role": message.role, "content": message.content]
            }
            var userInput = UserInput(messages: history, images: images)
            userInput.processing.resize = .init(width: 448, height: 448)
            let input = try await context.processor.prepare(input: userInput)
            try Task.checkCancellation()
            // Read the rendered template, not a model-name guess: thinking may be
            // disabled by a template flag or use a different protocol entirely.
            var primedReasoning = false
            if frameReasoning, let config = context.configuration.reasoningConfig,
               config.startDelimiter == "<think>", config.endDelimiter == "</think>" {
                let tail = input.text.tokens.asArray(Int.self).suffix(128)
                primedReasoning = ReasoningEventEmitter.promptEndsInsideReasoning(
                    renderedPromptTail: context.tokenizer.decode(tokenIds: Array(tail), skipSpecialTokens: false), config: config)
            }
            let promptTokens = input.text.tokens.size
            let parameters = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: topP)
            if useMTP && temperature == 0, let drafter {
                return try await generateMTP(input: input, context: context, drafter: drafter,
                    parameters: parameters, stop: stop, primedReasoning: primedReasoning, onUpdate: onUpdate)
            }
            let fallback = !useMTP ? "MTP is disabled" : temperature != 0 ? "Qwen MTP requires temperature 0" : "No supported local MTP sidecar"
            let (events, worker) = try MLXLMCommon.generateTokensTask(input: input, parameters: parameters, context: context)
            return try await withTaskCancellationHandler {
                var tokens: [Int] = []
                var info: GenerateCompletionInfo?
                var stoppedBySequence = false
                do {
                    for await event in events {
                        try Task.checkCancellation()
                        switch event {
                        case .token(let token):
                            tokens.append(token)
                            if !stop.isEmpty {
                                let text = context.tokenizer.decode(tokenIds: tokens)
                                if stop.contains(where: { text.contains($0) }) {
                                    stoppedBySequence = true
                                    worker.cancel()
                                    break
                                }
                            }
                            if tokens.count % 4 == 0 { onUpdate(AssistantMessageContent.framed(Self.visibleText(context.tokenizer.decode(tokenIds: tokens), stop: stop), primed: primedReasoning), tokens.count) }
                        case .info(let completion): info = completion
                        }
                        if stoppedBySequence { break }
                    }
                    // Wait for queued GPU work to settle before releasing this model's
                    // serial access or allowing a cancelled request's replacement to start.
                    await worker.value
                    try Task.checkCancellation()
                    var text = context.tokenizer.decode(tokenIds: tokens)
                    if let end = stop.compactMap({ text.range(of: $0)?.lowerBound }).min() { text = String(text[..<end]) }
                    return LocalChatResult(text: AssistantMessageContent.framed(text, primed: primedReasoning), tokensPerSecond: info?.tokensPerSecond ?? 0,
                        promptTokens: info?.promptTokenCount ?? promptTokens, completionTokens: tokens.count,
                        finishReason: !stoppedBySequence && info?.stopReason == .length ? "length" : "stop",
                        mtpFallbackReason: fallback, promptSeconds: info?.promptTime ?? 0,
                        generationSeconds: info?.generateTime ?? 0)
                } catch {
                    worker.cancel()
                    await worker.value
                    throw error
                }
            } onCancel: { worker.cancel() }
        }
    }

    private func generateMTP(input: LMInput, context: ModelContext, drafter: Qwen35MTPDraftModel,
                             parameters: GenerateParameters, stop: [String], primedReasoning: Bool,
                             onUpdate: @escaping @Sendable (String, Int) -> Void) async throws -> LocalChatResult {
        let iterator = try MTPSpeculativeTokenIterator(input: input, mainModel: context.model,
            drafter: drafter, parameters: parameters, blockSize: 2)
        // Create and join the worker even if cancellation arrived during prefill;
        // the cancellation handler below also runs when the task is already cancelled.
        // The generic public API retains MTP statistics and finalizes hybrid caches.
        // Unlike generateTokens(mtpDrafter:), it also exposes a worker we can join.
        let (events, worker) = MLXLMCommon.generateTask(promptTokenCount: input.text.tokens.size,
            modelConfiguration: context.configuration, tokenizer: context.tokenizer, iterator: iterator)
        return try await withTaskCancellationHandler {
            var text = ""
            var info: GenerateCompletionInfo?
            var stopped = false
            do {
                for await event in events {
                    try Task.checkCancellation()
                    switch event {
                    case .chunk(let chunk):
                        guard !stopped else { continue }
                        text += chunk
                        if let end = stop.compactMap({ text.range(of: $0)?.lowerBound }).min() {
                            text = String(text[..<end])
                            stopped = true
                            worker.cancel()
                        }
                        // Partial display counts are retokenized; final usage below
                        // comes from the worker's actual emitted-token ledger.
                        onUpdate(AssistantMessageContent.framed(Self.visibleText(text, stop: stop), primed: primedReasoning), context.tokenizer.encode(text: text).count)
                    case .info(let completion): info = completion
                    default: break
                    }
                }
                await worker.value
                try Task.checkCancellation()
                let count = info?.generationTokenCount ?? context.tokenizer.encode(text: text).count
                onUpdate(AssistantMessageContent.framed(text, primed: primedReasoning), count)
                return LocalChatResult(text: AssistantMessageContent.framed(text, primed: primedReasoning), tokensPerSecond: info?.tokensPerSecond ?? 0,
                    promptTokens: info?.promptTokenCount ?? input.text.tokens.size, completionTokens: count,
                    finishReason: !stopped && info?.stopReason == .length ? "length" : "stop",
                    proposedDraftTokens: info?.proposedDraftTokens, acceptedDraftTokens: info?.acceptedDraftTokens,
                    mtpFallbackReason: info?.passthroughReason, promptSeconds: info?.promptTime ?? 0,
                    generationSeconds: info?.generateTime ?? 0)
            } catch {
                worker.cancel()
                await worker.value
                throw error
            }
        } onCancel: { worker.cancel() }
    }

    // Hold a possible stop-sequence suffix until it is disambiguated by the
    // next chunk, so an HTTP stream never leaks a stop marker's partial prefix.
    nonisolated static func visibleText(_ text: String, stop: [String]) -> String {
        if let end = stop.compactMap({ text.range(of: $0)?.lowerBound }).min() {
            return String(text[..<end])
        }
        var held = 0
        for sequence in stop where sequence.count > 1 {
            let limit = min(text.count, sequence.count - 1)
            guard limit > held else { continue }
            for length in stride(from: limit, through: held + 1, by: -1) {
                if text.hasSuffix(sequence.prefix(length)) { held = length; break }
            }
        }
        return held == 0 ? text : String(text.dropLast(held))
    }

    enum ChatError: LocalizedError {
        case modelUnavailable(String), modelTooLarge(String), imagesUnsupported, invalidImage
        var errorDescription: String? {
            switch self {
            case .modelUnavailable(let id): "Local MLX model is unavailable: \(id). Select a complete supported model in HF_HOME."
            case .modelTooLarge(let id): "The weights for \(id) exceed the memory available for local inference. Select a smaller or quantized model."
            case .imagesUnsupported: "Select a local vision model to send images."
            case .invalidImage: "The attached image could not be decoded."
            }
        }
    }
}
