import Foundation
import ImageIO
import MLX
import Tokenizers
import Synchronization

enum Flux1Tests {
    static func run(_ directory: URL, cacheRoot: URL) async throws {
        let fixture = try MLX.loadArrays(url: directory.appendingPathComponent("fixtures.safetensors"))
        func get(_ name: String) -> MLXArray { fixture[name]! }
        func close(_ actual: MLXArray, _ expected: MLXArray, _ label: String, tolerance: Float = 0.0001) {
            precondition(actual.shape == expected.shape, label + " shape")
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
            print("\(label): max absolute error \(error)")
            precondition(error.isFinite && error < tolerance, label)
        }
        let tf = directory.appendingPathComponent("transformer")
        let transformer = try Flux1Transformer(weights: QwenImage21Weights(directory: tf),
            config: LocalImageModelCache.json(tf.appendingPathComponent("config.json")))
        for (guidance, label) in [(Float(3.5), "dit"), (Float(0), "dit_guidance_zero")] {
            close(try transformer(get("latent"), text: get("text"), pooled: get("pooled"),
                sigma: 0.73, guidance: guidance, ids: get("ids")), get(label), "FLUX.1 " + label)
        }
        let clip = try Flux1TextEncoder(directory: directory.appendingPathComponent("text_encoder"))
        close(try clip.clip(get("clip_tokens").asArray(Int.self), pooledIndex: 3), get("clip"), "CLIP pooled output")
        let t5 = try Flux1TextEncoder(directory: directory.appendingPathComponent("text_encoder_2"))
        close(try t5.t5(get("t5_tokens").asArray(Int.self)), get("t5"), "T5 no-scale attention and gated GELU")
        let vae = try Flux1VAE(directory: directory.appendingPathComponent("vae"))
        close(try vae.decode(get("vae_latent"), height: 32, width: 48), get("decoded"), "FLUX VAE unpack/scale/shift", tolerance: 0.0005)
        for count in [256, 1024, 4096] {
            close(try MLXArray(Flux1Pipeline.sigmas(steps: 4, tokens: count,
                config: LocalImageModelCache.json(directory.appendingPathComponent("scheduler/scheduler_config.json")))),
                get("sigmas_\(count)"), "FLUX schedule \(count)", tolerance: 1e-6)
        }
        let models = try LocalImageModelCache.scan(cacheRoot)
        guard let model = models.first(where: { $0.family == .flux1Dev }) else { fatalError("Complete FLUX.1-dev cache required") }
        let tests = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("token-tests.json"))) as! [[String: Any]]
        let tokenizer = try Flux1CLIPTokenizer(directory: model.directory.appendingPathComponent("tokenizer"))
        let t5Tokenizer = try await AutoTokenizer.from(modelFolder: model.directory.appendingPathComponent("tokenizer_2"))
        for test in tests {
            let prompt = test["prompt"] as! String
            let expected = test["tokens"] as! [Int]
            let actual: [Int]
            if test["component"] as! String == "tokenizer" { actual = try tokenizer.encode(prompt).0 }
            else {
                actual = try Flux1TextEncoder.t5Tokens(prompt, tokenizer: t5Tokenizer)
            }
            if actual != expected {
                print("TOKENIZER MISMATCH \(test["component"]!): \(prompt.debugDescription)")
                print("actual: \(actual.prefix(80)); expected: \(expected.prefix(80))")
            }
            precondition(actual == expected, "Tokenizer mismatch: \(test["component"]!) \(prompt)")
        }
        print("PASS: CLIP/T5 tokenization matches Transformers including Chinese, emoji, empty and truncated input")
        // Use an isolated shallow snapshot to verify missing shards/foreign
        // architectures without touching the actual HF cache.
        let copy = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: copy) }
        try FileManager.default.createDirectory(at: copy, withIntermediateDirectories: true)
        for subfolder in ["transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"] {
            let dst = copy.appendingPathComponent(subfolder)
            try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
            for file in try FileManager.default.contentsOfDirectory(at: model.directory.appendingPathComponent(subfolder), includingPropertiesForKeys: nil) {
                try FileManager.default.createSymbolicLink(at: dst.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
            }
        }
        // HF's source is a relative blob symlink; copying the symlink itself
        // would make it dangling outside the original snapshot hierarchy.
        try Data(contentsOf: model.directory.appendingPathComponent("model_index.json"))
            .write(to: copy.appendingPathComponent("model_index.json"))
        let complete = try LocalImageModelCache.scan(copy)
        precondition(complete.count == 1)
        let t5File = try LocalImageModelCache.shards(copy.appendingPathComponent("text_encoder_2"))[0]
        try FileManager.default.removeItem(at: t5File)
        let missing = try LocalImageModelCache.scan(copy)
        precondition(missing.isEmpty)
        for request in [LocalImageRequest(modelID: "x", prompt: "x", reference: Data([1])),
                        LocalImageRequest(modelID: "x", prompt: "x", reference: nil, control: Data([1])),
                        LocalImageRequest(modelID: "x", prompt: "x", reference: nil, guidance: .nan)] {
            do { try Flux1Pipeline.validate(request); preconditionFailure("Invalid FLUX.1 request accepted") }
            catch LocalImageError.invalid { }
        }
        print("PASS: native FLUX.1 numerical parity, complete cache discovery and request validation")
    }

    static func generate(cache: SpeechCacheAccess.Configuration, output: URL, size: Int) async throws {
        let service = LocalImageService()
        guard let model = try await service.models(cache: cache).first(where: { $0.family == .flux1Dev }) else { fatalError("FLUX.1-dev cache required") }
        let prompt = "Product photograph of a glossy red ceramic mug with a handle on the right, on a white table against a soft light gray background, soft studio lighting, centered composition, realistic, no text."
        var request = LocalImageRequest(modelID: model.id, prompt: prompt, reference: nil,
            width: size, height: size, steps: 28, seed: 42, trueCFGScale: Float(model.family.defaultTrueCFGScale))
        if let index = CommandLine.arguments.firstIndex(of: "--negative-prompt") {
            request.negativePrompt = CommandLine.arguments[index + 1]
        }
        if let index = CommandLine.arguments.firstIndex(of: "--steps") { request.steps = Int(CommandLine.arguments[index + 1])! }
        let result = try await service.generate(request, cache: cache) { label, _ in print(label) }
        let file = output.appendingPathComponent("flux1-dev-red-mug.png")
        try result.png.write(to: file)
        try LocalImageSmoke.checkImage(result.png, size: size)
        print("PASS: actual FLUX.1-dev \(size)x\(size), \(request.steps) steps, \(result.seconds)s: \(file.path)")
        let (started, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let task = Task {
            try await service.generate(.init(modelID: model.id, prompt: "A yellow tulip", reference: nil,
                width: 256, height: 256, steps: 2), cache: cache) { text, _ in
                    if text.contains("去噪") { continuation.yield(()) }
                }
        }
        var it = started.makeAsyncIterator()
        _ = await it.next(); task.cancel()
        do { _ = try await task.value; preconditionFailure("Cancellation returned an image") }
        catch is CancellationError { print("PASS: actual FLUX.1 denoising cancellation") }
        continuation.finish()
        let retry = try await service.generate(.init(modelID: model.id, prompt: "A yellow tulip", reference: nil,
            width: 256, height: 256, steps: 2), cache: cache) { _, _ in }
        try LocalImageSmoke.checkImage(retry.png, size: 256)
        print("PASS: actual FLUX.1 request after cancellation")
    }
}
