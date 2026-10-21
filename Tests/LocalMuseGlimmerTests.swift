import AppKit
import CoreImage
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import Synchronization

@main struct LocalMuseGlimmerTests {
    static func main() async throws {
        setbuf(stdout, nil)
        let registered = await VLMTypeRegistry.shared.contains("muse_glimmer")
        precondition(registered)
        let tokenizer = FixtureTokenizer()
        var output = try LocalMuseGlimmerOutput(tokenizer: tokenizer)
        func append(_ text: String) {
            for token in tokenizer.encode(text: text, addSpecialTokens: false) { output.append(token) }
        }
        append(" to=self")
        precondition(output.text(includeReasoning: true).isEmpty)
        output.append(200023)
        append("Think carefully.")
        precondition(AssistantMessageContent(output.text(includeReasoning: true)).isThinking)
        precondition(output.text(includeReasoning: false).isEmpty)
        output.append(200007)
        output.append(200022)
        append("assistant to=self")
        output.append(200023)
        append("Check again.")
        output.append(200007)
        output.append(200022)
        append("assistant to=weather.get")
        output.append(200023)
        append("<atem:function_calls>hidden tool payload</atem:function_calls>")
        output.append(200007)
        output.append(200022)
        append("assistant to=user")
        output.append(200023)
        // Literal marker-shaped text encoded as ordinary characters is not a frame.
        append("你好 🌍 — literal <|message|>.")
        output.append(200008)
        precondition(output.text(includeReasoning: false) == "你好 🌍 — literal <|message|>.")
        let parsed = AssistantMessageContent(output.text(includeReasoning: true))
        precondition(parsed.reasoning == "Think carefully.\nCheck again." && parsed.answer == output.text(includeReasoning: false))
        var plain = try LocalMuseGlimmerOutput(tokenizer: tokenizer)
        plain.append(200023)
        for token in tokenizer.encode(text: "OK", addSpecialTokens: false) { plain.append(token) }
        precondition(plain.text(includeReasoning: true) == "OK")
        do {
            _ = try LocalMuseGlimmerOutput(tokenizer: FixtureTokenizer(valid: false))
            fatalError("Invalid ATEM vocabulary accepted")
        } catch LocalMuseGlimmerOutput.OutputError.invalidControlToken {}
        print("PASS: token-aware ATEM primed/full headers, Unicode, reasoning, hidden tool frames, literal markers and invalid vocabulary")

        let configData = Data(#"""
        {"model_type":"muse_glimmer","image_token_id":15,"video_token_id":16,
         "out_hidden_size":32,"projector_hidden_size":32,
         "text_config":{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
          "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":128,
          "layer_types":["sliding_attention","full_attention"],"layer_rope_theta":[500000,0],
          "sliding_window":4,"max_position_embeddings":128},
         "vision_config":{"hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,
          "num_attention_heads":2,"patch_size":14,"patch_temporal":2,"merge_size":2,
          "pos_emb_height":4,"pos_emb_width":4,"max_position_embeddings":16,
          "layer_types":["full_attention"]}}
        """#.utf8)
        let config = try JSONDecoder().decode(MuseGlimmerConfiguration.self, from: configData)
        let reference = MuseGlimmer(config)
        quantize(model: reference) { _, module in
            guard (module is Linear || module is Embedding),
                  let weight = module.parameters().flattened().first(where: { $0.0 == "weight" })?.1,
                  weight.ndim == 2, weight.dim(1) % 32 == 0 else { return nil }
            return (groupSize: 32, bits: 8, mode: .mxfp8)
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("muse-native-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        precondition(weights.keys.contains { $0.hasSuffix(".scales") })
        try save(arrays: weights, metadata: ["format": "mlx"], url: directory.appendingPathComponent("model.safetensors"))
        let loaded = MuseGlimmer(config)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: Data(#"{"model_type":"muse_glimmer","quantization":{"group_size":32,"bits":8,"mode":"mxfp8"}}"#.utf8))
        try await loadWeights(modelDirectory: directory, model: loaded, perLayerQuantization: base.perLayerQuantization)
        let ids = MLXArray([1, 2, 3, 4, 5, 6])[.newAxis]
        let expected = reference(ids, cache: try reference.newCache(parameters: nil))
        let actual = loaded(ids, cache: try loaded.newCache(parameters: nil))
        let error = abs(actual - expected).max().item(Float.self)
        precondition(error.isFinite && error < 0.00001)
        let processor = MuseGlimmerProcessor(try JSONDecoder().decode(MuseGlimmerProcessorConfiguration.self,
            from: Data(#"{"image_processor":{"max_image_tokens":256}}"#.utf8)), tokenizer: tokenizer)
        var input = UserInput(prompt: "Describe this image", images: [.ciImage(solid(.red))])
        input.processing.resize = .init(width: 448, height: 448)
        let prepared = try await processor.prepare(input: input)
        let grid = prepared.image!.frames![0]
        precondition(prepared.text.tokens.asArray(Int.self).filter { $0 == 200092 }.count == grid.product / 4)
        precondition(grid.product / 4 <= 256 && grid.h % 2 == 0 && grid.w % 2 == 0)
        print("PASS: real Swift MLX MXFP8 checkpoint round-trip logits; 448-pixel vision preprocessing and patch/token alignment")
        try configData.write(to: directory.appendingPathComponent("config.json"))
        for name in ["tokenizer.json", "tokenizer_config.json", "processor_config.json"] {
            try Data("{}".utf8).write(to: directory.appendingPathComponent(name))
        }
        let discovered = try LocalChatModelCache.scan(in: directory, llmTypes: [], vlmTypes: ["muse_glimmer"])
        precondition(discovered.count == 1 && discovered[0].supportsImages)
        try FileManager.default.removeItem(at: directory.appendingPathComponent("processor_config.json"))
        let incomplete = try LocalChatModelCache.scan(in: directory, llmTypes: [], vlmTypes: ["muse_glimmer"])
        precondition(incomplete.isEmpty)
        print("PASS: Muse discovery accepts processor_config.json and rejects an incomplete vision checkpoint")
        if let path = ProcessInfo.processInfo.environment["SYNOMANCER_MUSE_TOKENIZER_DIR"] {
            let assets = URL(fileURLWithPath: path)
            let entry = LocalChatModel(id: "mlx-community/Muse-Glimmer-30B-mxfp8", directory: assets,
                modelType: "muse_glimmer", isVisionModel: true, supportsImages: true, weightBytes: 0)
            let loader = try await LocalChatTokenizerLoader.validated(for: entry)
            let actualTokenizer = try await loader.load(from: assets)
            let actualProcessor = MuseGlimmerProcessor(try JSONDecoder().decode(MuseGlimmerProcessorConfiguration.self,
                from: Data(contentsOf: assets.appendingPathComponent("processor_config.json"))), tokenizer: actualTokenizer)
            var prompt = UserInput(messages: [["role": "user", "content": [
                ["type": "text", "text": "What color is this image?"], ["type": "image"]]]],
                images: [.ciImage(solid(.red))])
            prompt.processing.resize = .init(width: 448, height: 448)
            let actualInput = try await actualProcessor.prepare(input: prompt)
            let actualTokens = actualInput.text.tokens.asArray(Int.self)
            precondition(actualTokenizer.decode(tokenIds: actualTokens).hasSuffix("<|start|>assistant"))
            precondition(actualTokens.filter { $0 == 200092 }.count == actualInput.image!.frames![0].product / 4)
            var actualOutput = try LocalMuseGlimmerOutput(tokenizer: actualTokenizer)
            let reply = " to=self<|message|>检查颜色。<|eom|><|start|>assistant to=user<|message|>red<|eot|>"
            for token in actualTokenizer.encode(text: reply, addSpecialTokens: false) { actualOutput.append(token) }
            precondition(actualOutput.text(includeReasoning: false) == "red")
            precondition(AssistantMessageContent(actualOutput.text(includeReasoning: true)).reasoning == "检查颜色。")
            print("PASS: actual Muse tokenizer, Swift Jinja image template, patch expansion and ATEM Unicode round-trip without loading 30B weights")
        }
        if CommandLine.arguments.contains("--components-only") { return }

        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let service = LocalChatService()
        let id = "mlx-community/Muse-Glimmer-30B-mxfp8"
        let models = try await service.models(cache: cache)
        guard models.contains(where: { $0.id == id && $0.isVisionModel && $0.supportsImages }) else {
            throw NSError(domain: "MuseNativeTest", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Muse vision model missing from discovered cache: \(models)"])
        }
        print("START: real Muse native history generation")
        let updates = Mutex<[(String, Int)]>([])
        let history: [LocalChatMessage] = [
            .init(role: "user", content: "The passphrase is REDWOOD-472."),
            .init(role: "assistant", content: "I will remember it."),
            .init(role: "user", content: "Reply only with the exact passphrase I gave you.")]
        // The published template defaults to high reasoning. Its hidden tokens
        // share this budget with the answer, so leave room for both.
        let result = try await service.generate(modelID: id, messages: history, cache: cache,
            temperature: 0, maxTokens: 1024, frameReasoning: true) { text, count in updates.withLock { $0.append((text, count)) } }
        let answer = AssistantMessageContent(result.text).answer
        try require(answer.contains("REDWOOD-472") && result.finishReason == "stop",
            "History failed: tokens=\(result.completionTokens), finish=\(result.finishReason), text=\(result.text.debugDescription)")
        try require(!result.text.contains("<|message|>") && !result.text.contains("to=user"), "ATEM headers leaked")
        try require(result.tokensPerSecond > 0 && result.completionTokens > 0, "Missing real generation usage")
        try require(updates.withLock { $0.last?.0 == result.text && $0.last?.1 == result.completionTokens }, "Final stream and result usage differ")
        print("PASS: real MXFP8 native history, ATEM display and final stream usage; \(result.tokensPerSecond) tokens/s; \(result.completionTokens) tokens; \(answer.debugDescription)")
        let followup = try await service.generate(modelID: id, messages: history + [
            .init(role: "assistant", content: result.text),
            .init(role: "user", content: "Reply only with the three digits at the end of that passphrase.")],
            cache: cache, temperature: 0, maxTokens: 1024) { _, _ in }
        try require(followup.text.trimmingCharacters(in: .whitespacesAndNewlines) == "472",
            "Generated assistant history replay failed: \(followup.text.debugDescription)")
        print("PASS: real follow-up replays the generated assistant answer: \(followup.text.debugDescription)")
        for (color, name) in [(CIColor.red, "red"), (CIColor.blue, "blue")] {
            let image = CIContext().createCGImage(solid(color), from: solid(color).extent)!
            let bitmap = NSBitmapImageRep(cgImage: image)
            let data = bitmap.representation(using: .png, properties: [:])!
            let response = try await service.generate(modelID: id,
                messages: [.init(role: "user", content: "What is the dominant color of this image? Reply with one English color word only.", image: data)],
                cache: cache, temperature: 0, maxTokens: 1024) { _, _ in }
            try require(response.text.lowercased().contains(name), "Vision missed \(name): \(response.text)")
            print("PASS: real image input identifies \(name): \(response.text.debugDescription)")
        }
        if let reference = ProcessInfo.processInfo.environment["SYNOMANCER_MUSE_IMAGE"] {
            let response = try await service.generate(modelID: id,
                messages: [.init(role: "user", content: "Describe this image in one short English sentence.",
                    image: Data(contentsOf: URL(fileURLWithPath: reference)))],
                cache: cache, temperature: 0, maxTokens: 1024) { _, _ in }
            let caption = response.text.lowercased()
            try require(caption.contains("red") && (caption.contains("mug") || caption.contains("cup")),
                "Photo fixture missed the visible red mug: \(response.text)")
            print("PASS: actual red-mug image caption through native vision: \(response.text.debugDescription)")
        }
        let stopMessages = [LocalChatMessage(role: "user", content: "Reply exactly: HELLO WORLD")]
        let full = try await service.generate(modelID: id, messages: stopMessages, cache: cache,
            temperature: 0, maxTokens: 1024) { _, _ in }
        try require(full.text.contains("WORLD"), "Stop fixture must actually generate its stop string: \(full.text.debugDescription)")
        let stopped = try await service.generate(modelID: id,
            messages: stopMessages, cache: cache,
            temperature: 0, maxTokens: 1024, stop: ["WORLD"]) { _, _ in }
        try require(stopped.text == String(full.text[..<full.text.range(of: "WORLD")!.lowerBound])
            && stopped.finishReason == "stop", "Stop output differs from the baseline prefix")
        let (started, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let task = Task {
            try await service.generate(modelID: id, messages: [.init(role: "user", content: "Write a long story about the ocean.")],
                cache: cache, temperature: 0, maxTokens: 1024) { _, _ in continuation.yield(()) }
        }
        var next = started.makeAsyncIterator()
        _ = await next.next()
        task.cancel()
        do { _ = try await task.value; fatalError("Cancellation ignored") } catch is CancellationError {}
        continuation.finish()
        let restarted = try await service.generate(modelID: id, messages: [.init(role: "user", content: "Reply only with OK.")],
            cache: cache, temperature: 0, maxTokens: 1024) { _, _ in }
        try require(restarted.text.contains("OK"), "Restart failed: \(restarted.text.debugDescription)")
        print("PASS: real ATEM stop strings, cancellation and immediate native model reuse")
    }

    static func require(_ passed: Bool, _ message: String) throws {
        guard passed else {
            throw NSError(domain: "MuseNativeTest", code: 2,
                userInfo: [NSLocalizedDescriptionKey: message])
        }
    }

    static func solid(_ color: CIColor) -> CIImage {
        CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 512, height: 256))
    }
}

private struct FixtureTokenizer: MLXLMCommon.Tokenizer {
    var valid = true
    let controls = ["<|start|>":200022, "<|message|>":200023, "<|eom|>":200007,
        "<|eot|>":200008, "<|end_of_text|>":200001, "<|patch|>":200092,
        "<|image_start|>":200080, "<|image_end|>":200081]
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let id = convertTokenToId(text) { return [id] }
        return text.unicodeScalars.map { Int($0.value) + 1_000_000 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            if let control = convertIdToToken(id) { return skipSpecialTokens ? "" : control }
            return UnicodeScalar(id - 1_000_000).map(String.init) ?? ""
        }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { valid ? controls[token] : nil }
    func convertIdToToken(_ id: Int) -> String? { controls.first { $0.value == id }?.key }
    var bosToken: String? { nil }
    var eosToken: String? { "<|eot|>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        [1, 200092, 2]
    }
}
