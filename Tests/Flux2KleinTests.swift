import Foundation
import MLX

enum Flux2KleinTests {
    static func run(_ directory: URL) throws {
        let fixture = try MLX.loadArrays(url: directory.appendingPathComponent("fixtures.safetensors"))
        func get(_ name: String) -> MLXArray { fixture[name]! }
        func close(_ actual: MLXArray, _ expected: MLXArray, _ label: String, tolerance: Float = 0.0002) {
            precondition(actual.shape == expected.shape, label + " shape")
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
            print("\(label): max absolute error \(error)")
            precondition(error.isFinite && error < tolerance, label)
        }
        let textFolder = directory.appendingPathComponent("text_encoder")
        let encoder = try Flux2KleinTextEncoder(weights: QwenImage21Weights(directory: textFolder),
            config: LocalImageModelCache.json(textFolder.appendingPathComponent("config.json")))
        close(try encoder.encode(get("text_tokens").asType(.int32).asArray(Int32.self).map(Int.init), validCount: 4),
            get("text_features"), "Qwen3 hidden layers 9/18/27 with right padding")
        let folder = directory.appendingPathComponent("transformer")
        let transformer = try Flux2KleinTransformer(weights: QwenImage21Weights(directory: folder),
            config: LocalImageModelCache.json(folder.appendingPathComponent("config.json")))
        close(try transformer(get("latent"), text: get("text"), sigma: 0.73, ids: get("ids")), get("dit"), "Klein double/single blocks")
        close(try transformer(concatenated([get("latent"), get("reference")], axis: 1), text: get("text"),
            sigma: 0.73, ids: get("edit_ids")), get("edit"), "Klein reference attention")
        // Reproduce the full FP16 export's range failure: text features are
        // finite, but context projection exceeds FP16's maximum finite value.
        let half = QwenImage21Weights(arrays: transformer.weights.arrays.mapValues { $0.asType(.float16) })
        half.arrays["context_embedder.weight"] = MLXArray.ones(try half.get("context_embedder.weight").shape, dtype: .float16)
        let rangeModel = try Flux2KleinTransformer(weights: half,
            config: LocalImageModelCache.json(folder.appendingPathComponent("config.json")))
        let computeType = try half.get("context_embedder.weight").dtype
        let largeText = MLXArray.full(get("text").shape, values: MLXArray(Float(22560)), dtype: computeType)
        let prediction = try rangeModel(get("latent").asType(computeType), text: largeText,
            sigma: 0.73, ids: get("ids"))
        precondition(all(isFinite(prediction)).item(Bool.self), "FP16 export context overflow")
        print("PASS: large finite context from FP16 export produces finite Klein prediction")
        let vae = try Flux2KleinVAE(directory: directory.appendingPathComponent("vae"))
        close(try vae.encode(get("rgb")), get("encoded"), "Klein VAE encode, patchify, BN")
        close(try vae.decode(get("encoded"), height: 32, width: 48), get("decoded"), "Klein VAE BN, unpatchify, decode", tolerance: 0.001)
        for count in [256, 4096, 5000] {
            close(MLXArray(Flux2KleinPipeline.sigmas(steps: 4, tokens: count)), get("sigmas_\(count)"), "Klein schedule \(count)", tolerance: 1e-6)
        }
        let portrait = try Flux2KleinPipeline.referenceSize(width: 512, height: 768)
        let wide = try Flux2KleinPipeline.referenceSize(width: 2048, height: 1024)
        precondition(portrait == (512, 768))
        precondition(wide == (1440, 720))
        do {
            _ = try Flux2KleinPipeline.referenceSize(width: 15, height: 100)
            preconditionFailure("Tiny reference must fail")
        } catch LocalImageError.invalid { }
        try discovery(directory)
        print("PASS: FLUX.2 Klein CPU Diffusers parity, discovery and reference dimensions")
    }

    static func discovery(_ fixtures: URL) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("models--mlx-community--FLUX.2-klein-9B/snapshots/test")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        func json(_ value: [String: Any], _ path: String) throws {
            let url = model.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: value).write(to: url)
        }
        for component in ["transformer", "text_encoder", "vae"] {
            try FileManager.default.copyItem(at: fixtures.appendingPathComponent(component), to: model.appendingPathComponent(component))
        }
        try json(["_class_name": "Flux2KleinPipeline", "is_distilled": true], "model_index.json")
        try json(["joint_attention_dim": 12288, "num_layers": 8, "num_single_layers": 24,
                  "guidance_embeds": false], "transformer/config.json")
        try json([:], "tokenizer/tokenizer.json")
        try json([:], "tokenizer/tokenizer_config.json")
        try json([:], "scheduler/scheduler_config.json")
        let models = try LocalImageModelCache.scan(root)
        precondition(models.count == 1 && models[0].family == .flux2Klein9B)
        precondition(models[0].family.defaultSteps == 4 && !models[0].family.supportsControl
                     && models[0].family.supportsReference)
        let direct = try LocalImageModelCache.scan(model)
        precondition(direct.count == 1)
        try json(["_class_name": "Flux2KleinPipeline", "is_distilled": false], "model_index.json")
        let base = try LocalImageModelCache.scan(root)
        precondition(base.isEmpty, "Base checkpoint must not run distilled pipeline")
        try json(["_class_name": "Flux2KleinPipeline", "is_distilled": true], "model_index.json")
        try json(["weight_map": ["x": "missing.safetensors"]], "transformer/model.safetensors.index.json")
        let incomplete = try LocalImageModelCache.scan(root)
        precondition(incomplete.isEmpty, "Incomplete snapshot must not be listed")
    }

    static func checks(cache: SpeechCacheAccess.Configuration) async throws {
        let service = LocalImageService()
        guard let model = try await service.models(cache: cache).first(where: {
            $0.id == "mlx-community/FLUX.2-klein-9B" && $0.family == .flux2Klein9B
        }) else { throw LocalImageError.missing("mlx-community/FLUX.2-klein-9B") }
        var request = LocalImageRequest(modelID: model.id, prompt: "A yellow tulip in a white vase",
            width: 256, height: 256, steps: model.family.defaultSteps, seed: 42)
        request.control = Data([1])
        do {
            _ = try await service.generate(request, cache: cache) { _, _ in }
            preconditionFailure("Klein must reject Qwen ControlNet")
        } catch LocalImageError.invalid { print("PASS: Klein rejects unsupported ControlNet") }
        request.control = nil
        let pending = request
        let (started, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let task = Task {
            // Close the stream on early failure as well, so the observer cannot hang.
            defer { continuation.finish() }
            return try await service.generate(pending, cache: cache) { stage, _ in
                if stage.hasPrefix("Klein 去噪 2/") { continuation.yield(()) }
            }
        }
        var iterator = started.makeAsyncIterator()
        guard await iterator.next() != nil else {
            _ = try await task.value
            throw LocalImageError.invalid("Klein did not reach its second denoising step")
        }
        task.cancel()
        do { _ = try await task.value; preconditionFailure("Cancellation returned a Klein image") }
        catch is CancellationError { print("PASS: actual Klein cancellation after completed first denoising step") }
        let retry = try await service.generate(request, cache: cache) { stage, _ in print(stage) }
        try LocalImageSmoke.checkImage(retry.png, size: 256)
        print("PASS: actual Klein request after cancellation, \(retry.seconds)s")
    }
}
