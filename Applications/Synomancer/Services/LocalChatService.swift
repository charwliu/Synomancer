import Foundation
@preconcurrency import CoreImage
import MLX
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
    var cachedPromptTokens: Int = 0
}

actor LocalChatService {
    static let shared = LocalChatService()
    // All model access is actor-owned and protected by the generating gate across
    // suspension points. The upstream worker is joined before another request runs.
    private var context: ModelContext?
    private var drafter: Qwen35MTPDraftModel?
    private var loadedModel: LocalChatModel?
    private var loadedDirectory: URL?
    private var loadedCache: SpeechCacheAccess.Configuration?
    private var generating = false
    private struct PromptCheckpoint {
        let tokens: [Int]
        let cache: [KVCache]
        let state: LMOutput.State?
    }
    // One pre-decode checkpoint: never trim recurrent state or reuse generated
    // reasoning that a chat template may remove when rendering the next turn.
    private var promptCheckpoint: PromptCheckpoint?

    func clearPromptCache() {
        promptCheckpoint = nil
    }

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
        let discovered = try LocalChatModelCache.scan(in: root, llmTypes: llmTypes, vlmTypes: vlmTypes)
        // Explicit Refresh still notices a new/deleted snapshot. Do not replace
        // the running model here; the next serial request handles that safely.
        if loadedCache == cache, let previous = loadedModel {
            loadedModel = discovered.first { $0.id == previous.id }
        }
        return discovered
    }

    func generate(modelID: String, messages: [LocalChatMessage], cache: SpeechCacheAccess.Configuration,
                  temperature: Float = 0.6, maxTokens: Int = 2048, topP: Float = 1, stop: [String] = [], useMTP: Bool = true, frameReasoning: Bool = false, reusePromptCache: Bool = true, prefillStepSize: Int? = nil,
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
        let model: LocalChatModel
        if let loadedModel, context != nil, loadedCache == cache, loadedModel.id == modelID,
           FileManager.default.fileExists(atPath: loadedModel.directory.path) {
            model = loadedModel
        } else {
            guard let found = try await models(cache: cache).first(where: { $0.id == modelID }) else {
                throw ChatError.modelUnavailable(modelID)
            }
            model = found
        }
        guard model.weightBytes < ProcessInfo.processInfo.physicalMemory * 4 / 5 else {
            throw ChatError.modelTooLarge(modelID)
        }
        if messages.contains(where: { $0.image != nil }) && !model.supportsImages { throw ChatError.imagesUnsupported }
        if context == nil || loadedDirectory != model.directory || loadedCache != cache {
            let configData = try Data(contentsOf: model.directory.appendingPathComponent("config.json"))
            if let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] {
                try LocalQwenMTP.validateGating(config)
            }
            let tokenizerLoader = try await LocalChatTokenizerLoader.validated(for: model)
            try Task.checkCancellation()
            promptCheckpoint = nil
            context = nil
            drafter = nil
            if model.isVisionModel {
                let factory = model.modelType == "qwen3_5_moe" ? LocalQwenMoEModel.factory : VLMModelFactory.shared
                context = try await factory.load(from: model.directory, using: tokenizerLoader)
            } else {
                context = try await LLMModelFactory.shared.load(from: model.directory, using: tokenizerLoader)
            }
            loadedModel = model
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
            var parameters = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: topP)
            parameters.prefill.stepSize = prefillStepSize
            if useMTP && temperature == 0, let drafter {
                promptCheckpoint = nil
                return try await generateMTP(input: input, context: context, drafter: drafter,
                    parameters: parameters, stop: stop, primedReasoning: primedReasoning, onUpdate: onUpdate)
            }
            let fallback = !useMTP ? "MTP is disabled" : temperature != 0 ? "Qwen MTP requires temperature 0" : "No supported local MTP sidecar"
            // Restrict reuse to verified text paths. Media carries additional
            // positional state and must continue through the full prepared input.
            let canCache = reusePromptCache && images.isEmpty && input.text.tokens.ndim == 1
                && input.text.mask == nil
                && ["qwen3_5", "qwen3_5_text", "gemma4", "gemma4_text"].contains(model.modelType)
            let promptIDs = canCache ? input.text.tokens.asArray(Int.self) : []
            var cachedCount = 0
            var generationInput = input
            var generationCache: [KVCache]?
            var generationState: LMOutput.State?
            if canCache, let checkpoint = promptCheckpoint,
               promptIDs.count > checkpoint.tokens.count,
               promptIDs.starts(with: checkpoint.tokens) {
                cachedCount = checkpoint.tokens.count
                generationInput = LMInput(tokens: MLXArray(Array(promptIDs.dropFirst(cachedCount))))
                generationCache = checkpoint.cache.map { $0.copy() }
                generationState = checkpoint.state
            }
            // A mismatch in the retained prefix always rebuilds. The suffix is
            // recomputed, including changes to recent turns and template markers.
            if cachedCount == 0 { promptCheckpoint = nil }
            if canCache && generationCache == nil {
                generationCache = try context.model.newCache(parameters: parameters)
            }
            // Keep the unstable assistant/template tail out of the checkpoint.
            // A fixed token boundary also permits retries and short follow-ups to
            // reuse the same prefix without rolling back Qwen's recurrent state.
            // Retain Qwen's validated boundaries: tighter checkpoints changed
            // the answer in the real hybrid-attention regression fixture.
            let checkpointStride = model.modelType.hasPrefix("gemma4") ? 128 : 256
            let checkpointTail = model.modelType.hasPrefix("gemma4") ? 32 : 128
            let checkpointCount = canCache && promptTokens >= 384
                ? (promptTokens - checkpointTail) / checkpointStride * checkpointStride : 0
            var checkpointSeconds: Double = 0
            if checkpointCount > cachedCount, let generationCache {
                let start = ContinuousClock.now
                let prefix = LMInput(tokens: MLXArray(Array(promptIDs[cachedCount..<checkpointCount])))
                // Only prefill this prefix. The sampled token is discarded and is
                // never committed to cache; greedy sampling does not consume RNG.
                var prefixParameters = GenerateParameters(maxTokens: 1, temperature: 0)
                prefixParameters.prefill = parameters.prefill
                let prefixIterator = try TokenIterator(input: prefix, model: context.model,
                    cache: generationCache, state: generationState,
                    parameters: prefixParameters)
                generationState = prefixIterator.state
                let bytes = generationCache.flatMap { $0.state }.reduce(0) { $0 + $1.nbytes }
                let limit = min(UInt64(2 * 1024 * 1024 * 1024), ProcessInfo.processInfo.physicalMemory / 64)
                if bytes > 0 && UInt64(bytes) <= limit {
                    let snapshot = generationCache.map { $0.copy() }
                    eval(snapshot.flatMap { $0.state })
                    promptCheckpoint = PromptCheckpoint(tokens: Array(promptIDs.prefix(checkpointCount)),
                        cache: snapshot, state: generationState)
                }
                generationInput = LMInput(tokens: MLXArray(Array(promptIDs.dropFirst(checkpointCount))))
                let elapsed = start.duration(to: .now).components
                checkpointSeconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            }
            let iterator = try TokenIterator(input: generationInput, model: context.model,
                cache: generationCache, state: generationState, parameters: parameters)
            let (events, worker) = MLXLMCommon.generateTokenTask(promptTokenCount: promptTokens,
                modelConfiguration: context.configuration, tokenizer: context.tokenizer, iterator: iterator)
            return try await withTaskCancellationHandler {
                var tokens: [Int] = []
                var info: GenerateCompletionInfo?
                var stoppedBySequence = false
                var lastUpdate: ContinuousClock.Instant?
                var lastPublishedCount = 0
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
                var streamedText = ""
                do {
                    for await event in events {
                        try Task.checkCancellation()
                        switch event {
                        case .token(let token):
                            tokens.append(token)
                            if stop.isEmpty { detokenizer.append(token: token) }
                            var decoded: String?
                            if !stop.isEmpty {
                                decoded = context.tokenizer.decode(tokenIds: tokens)
                                if stop.contains(where: { decoded!.contains($0) }) {
                                    stoppedBySequence = true
                                    worker.cancel()
                                    break
                                }
                            }
                            let now = ContinuousClock.now
                            if lastUpdate == nil || (tokens.count - lastPublishedCount >= 4
                                && lastUpdate!.duration(to: now) >= .milliseconds(100)) {
                                // Upstream detokenization retains only the current
                                // newline-delimited segment, including UTF-8 boundaries.
                                // Stop strings retain full decoding for exact matching.
                                if stop.isEmpty {
                                    if let chunk = detokenizer.next() { streamedText += chunk }
                                    decoded = streamedText
                                }
                                onUpdate(AssistantMessageContent.framed(Self.visibleText(decoded!, stop: stop), primed: primedReasoning), tokens.count)
                                lastUpdate = now
                                lastPublishedCount = tokens.count
                            }
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
                    onUpdate(AssistantMessageContent.framed(text, primed: primedReasoning), tokens.count)
                    return LocalChatResult(text: AssistantMessageContent.framed(text, primed: primedReasoning), tokensPerSecond: info?.tokensPerSecond ?? 0,
                        promptTokens: info?.promptTokenCount ?? promptTokens, completionTokens: tokens.count,
                        finishReason: !stoppedBySequence && info?.stopReason == .length ? "length" : "stop",
                        mtpFallbackReason: fallback, promptSeconds: (info?.promptTime ?? 0) + checkpointSeconds,
                        generationSeconds: info?.generateTime ?? 0, cachedPromptTokens: cachedCount)
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
            var lastUpdate: ContinuousClock.Instant?
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
                        let now = ContinuousClock.now
                        if lastUpdate == nil || lastUpdate!.duration(to: now) >= .milliseconds(100) {
                            onUpdate(AssistantMessageContent.framed(Self.visibleText(text, stop: stop), primed: primedReasoning), context.tokenizer.encode(text: text).count)
                            lastUpdate = now
                        }
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
