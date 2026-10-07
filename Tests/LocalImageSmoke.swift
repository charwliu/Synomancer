import Foundation
import ImageIO
import MLX
import Synchronization
@preconcurrency import CoreImage

@main
struct LocalImageSmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        if CommandLine.arguments.contains("--parity") {
            try parity(URL(fileURLWithPath: CommandLine.arguments[2]))
            return
        }
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let output = URL(fileURLWithPath: CommandLine.arguments[2])
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let service = LocalImageService()
        let models = try await service.models(cache: cache)
        guard let model = models.first(where: { $0.quantized }) else { fatalError("A complete native MLX Qwen-Image-2.1 cache is required") }
        if CommandLine.arguments.contains("--prefix-benchmark") {
            try await prefixBenchmark(cache: cache, model: model)
            return
        }
        if CommandLine.arguments.contains("--checks") {
            try await checks(service, cache: cache, model: model)
            return
        }
        let sizeIndex = CommandLine.arguments.firstIndex(of: "--size")
        let size = sizeIndex.flatMap { Int(CommandLine.arguments[$0 + 1]) } ?? 512
        let suffix = size == 512 ? "" : "-\(size)"
        print("MODEL \(model.id), \(Double(model.bytes) / 1e9) GB")
        var request = LocalImageRequest(modelID: model.id,
            prompt: "A studio photograph of a bright red ceramic coffee mug on a plain light gray table, soft daylight, photorealistic, no text.",
            width: size, height: size, steps: 40, seed: 42)
        if let index = CommandLine.arguments.firstIndex(of: "--reference") {
            request.reference = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[index + 1]))
            request.prompt = "Change the bright red ceramic coffee mug to a vivid blue ceramic coffee mug. Keep the mug shape, handle, table, background, lighting and camera angle unchanged."
            if let index = CommandLine.arguments.firstIndex(of: "--prompt") { request.prompt = CommandLine.arguments[index + 1] }
            request.seed = 43
            let edited = try await service.generate(request, cache: cache) { stage, _ in print("\(Date().timeIntervalSince1970) \(stage)") }
            try edited.png.write(to: output.appendingPathComponent("reference-edit-\(size).png"))
            try checkImage(edited.png, size: size)
            print("PASS: actual native existing-reference editing PNG, \(edited.seconds)s")
            return
        }
        let generated = try await service.generate(request, cache: cache) { stage, _ in print("\(Date().timeIntervalSince1970) \(stage)") }
        try generated.png.write(to: output.appendingPathComponent("generated-red-mug\(suffix).png"))
        try checkImage(generated.png, size: size)
        print("PASS: actual native text-to-image PNG, \(generated.seconds)s")
        request.reference = generated.png
        request.prompt = "Change the bright red ceramic coffee mug to a vivid blue ceramic coffee mug. Keep the mug shape, handle, table, background, lighting and camera angle unchanged."
        request.seed = 43
        let edited = try await service.generate(request, cache: cache) { stage, _ in print("\(Date().timeIntervalSince1970) \(stage)") }
        try edited.png.write(to: output.appendingPathComponent("edited-blue-mug\(suffix).png"))
        try checkImage(edited.png, size: size)
        precondition(edited.png != generated.png)
        print("PASS: actual native single-reference editing PNG, \(edited.seconds)s")
        if CommandLine.arguments.contains("--repeat") {
            let stages = Mutex<[String]>([])
            let repeated = try await service.generate(request, cache: cache) { stage, _ in
                stages.withLock { $0.append(stage) }
                print("\(Date().timeIntervalSince1970) \(stage)")
            }
            precondition(stages.withLock { $0.contains("复用文本和参考图去噪缓存…") })
            try repeated.png.write(to: output.appendingPathComponent("edited-blue-mug-repeat\(suffix).png"))
            try compareImage(repeated.png, edited.png, label: "prefix reuse 40 steps")
            print("PASS: repeated edit matches within pixel tolerance, \(repeated.seconds)s")
            // Changing both seed and step count must still reuse an independent
            // prefix, and match a fresh uncached run of those same parameters.
            var changed = request
            changed.steps = 2; changed.seed += 1
            let reused = try await service.generate(changed, cache: cache) { _, _ in }
            await service.clearSessionCache()
            let freshStages = Mutex<[String]>([])
            let fresh = try await service.generate(changed, cache: cache) { stage, _ in
                freshStages.withLock { $0.append(stage) }
            }
            try compareImage(reused.png, fresh.png, label: "new seed/steps vs uncached")
            precondition(freshStages.withLock { !$0.contains(where: { $0.hasPrefix("复用") }) })
            print("PASS: new seed/steps match uncached image tolerance; clearSessionCache forces fresh encoding")
        }
        print("MLX peak allocated bytes: \(Memory.peakMemory)")
    }

    static func prefixBenchmark(cache: SpeechCacheAccess.Configuration, model: LocalImageModel) async throws {
        guard let index = CommandLine.arguments.firstIndex(of: "--reference") else {
            fatalError("--prefix-benchmark requires --reference")
        }
        let reference = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[index + 1]))
        let baseline = LocalImageService(prefixCacheLimit: 0)
        let optimized = LocalImageService()
        var request = LocalImageRequest(modelID: model.id,
            prompt: "Change the bright red ceramic coffee mug to a vivid blue ceramic coffee mug. Keep the mug shape, handle, table, background, lighting and camera angle unchanged.",
            reference: reference, width: 512, height: 512, steps: 2, seed: 43)
        // Both services retain identical text/image conditioning. Only the
        // bounded prefix cache differs; warm-up is excluded from the timings.
        _ = try await baseline.generate(request, cache: cache) { _, _ in }
        _ = try await optimized.generate(request, cache: cache) { _, _ in }
        request.steps = 8
        var expected: Data?
        var deterministic: [Bool: Data] = [:]
        for (trial, usePrefix) in [false, true, true, false].enumerated() {
            let service = usePrefix ? optimized : baseline
            let stages = Mutex<[String]>([])
            let result = try await service.generate(request, cache: cache) { stage, _ in
                stages.withLock { $0.append(stage) }
            }
            precondition(stages.withLock { $0.contains("复用文本和参考图去噪缓存…") } == usePrefix)
            if let expected { try compareImage(result.png, expected, label: "prefix benchmark") }
            else { expected = result.png }
            if let previous = deterministic[usePrefix] { precondition(previous == result.png) }
            deterministic[usePrefix] = result.png
            print("BENCH trial=\(trial) prefix=\(usePrefix) seconds=\(result.seconds) pixelTolerancePassed=true")
        }
        await baseline.clearSessionCache()
        await optimized.clearSessionCache()
    }

    // Different GEMM sequence lengths may round BF16 differently. Keep a
    // strict image bound: RMS <= 1 level and mean absolute error <= 0.25 on
    // the 0...255 RGBA scale, rather than requiring identical PNG bytes.
    static func compareImage(_ actual: Data, _ expected: Data, label: String) throws {
        let context = CIContext(options: [.cacheIntermediates: false])
        func pixels(_ data: Data) -> (CGRect, [UInt8]) {
            let image = CIImage(data: data)!
            let rect = image.extent
            var result = [UInt8](repeating: 0, count: Int(rect.width * rect.height) * 4)
            context.render(image, toBitmap: &result, rowBytes: Int(rect.width) * 4,
                bounds: rect, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
            return (rect, result)
        }
        let (shape, a) = pixels(actual), (expectedShape, b) = pixels(expected)
        precondition(shape == expectedShape)
        var squared = 0.0, absolute = 0.0
        for (x, y) in zip(a, b) {
            let delta = Double(x) - Double(y)
            squared += delta * delta; absolute += abs(delta)
        }
        let rms = sqrt(squared / Double(a.count)), mean = absolute / Double(a.count)
        print("PIXELS \(label): rms=\(rms), meanAbsolute=\(mean), exact=\(actual == expected)")
        precondition(rms <= 1 && mean <= 0.25, "Image difference exceeds rounding tolerance")
    }

    static func checkImage(_ png: Data, size: Int = 512) throws {
        guard let source = CGImageSourceCreateWithData(png as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { fatalError("Generated output is not an image") }
        precondition(image.width == size && image.height == size && png.count > 10000)
    }

    static func parity(_ folder: URL) throws {
        func close(_ actual: MLXArray, _ expected: MLXArray, _ label: String, tolerance: Float = 0.00003) {
            precondition(actual.shape == expected.shape, "\(label) shape mismatch")
            let error = abs(actual - expected).max().item(Float.self)
            print("\(label) maximum absolute error: \(error)")
            precondition(error < tolerance, "\(label) differs from Diffusers")
        }
        let weights = try QwenImage21Weights(directory: folder.appendingPathComponent("transformer"))
        let config = try LocalImageModelCache.json(folder.appendingPathComponent("transformer/config.json"))
        let transformer = try QwenImage21Transformer(weights: weights, config: config)
        let tensors = try MLX.loadArrays(url: folder.appendingPathComponent("dit-inputs.safetensors"))
        let layout = try QwenImage21Layout(imageSlots: [false, true, false],
            shapes: [(2, 2), (4, 4)], axes: transformer.axes)
        var cache: [(MLXArray, MLXArray)] = []
        let first = try transformer.predict(latents: tensors["latents"]!, text: tensors["text"]!,
            timestep: 0.75, layout: layout, cache: &cache)
        close(first, tensors["first"]!, "reference DiT first pass")
        precondition(cache.count == 2 && cache[0].0.dim(2) == layout.prefix)
        let cached = try transformer.predict(latents: tensors["latents"]!, text: tensors["text"]!,
            timestep: 0.25, layout: layout, cache: &cache)
        close(cached, tensors["cached"]!, "reference DiT cached pass")
        close(cached, tensors["uncached"]!, "cached vs full reference")
        let vae = try QwenImage21VAE(weights: QwenImage21Weights(directory: folder.appendingPathComponent("vae")),
            config: LocalImageModelCache.json(folder.appendingPathComponent("vae/config.json")))
        let images = try MLX.loadArrays(url: folder.appendingPathComponent("vae-inputs.safetensors"))
        let source = images["image"]![0..., 0..., 0, 0..., 0...].transposed(0, 2, 3, 1)
        let encoded = try vae.encode(source)
        let expected = images["encoded"]![0..., 0..., 0, 0..., 0...].transposed(0, 2, 3, 1)
        close(encoded, expected, "RGBA residual VAE encoder")
        let decoded = try vae.decode(encoded)
        close(decoded, images["decoded"]![0..., 0..., 0, 0..., 0...].transposed(0, 2, 3, 1),
            "RGBA residual VAE decoder", tolerance: 0.0001)
        let sigmas = try LocalImageService.sigmas(steps: 40, tokens: 1024,
            config: LocalImageModelCache.json(folder.appendingPathComponent("scheduler.json")))
        close(MLXArray(sigmas), try MLX.loadArrays(url: folder.appendingPathComponent("scheduler.safetensors"))["sigmas"]!,
            "FlowMatch Euler sigmas", tolerance: 0.000001)
        let precision = try MLX.loadArrays(url: folder.appendingPathComponent("precision.safetensors"))
        for (name, dtype) in [("bf16", DType.bfloat16), ("fp16", .float16), ("fp32", .float32)] {
            for (index, sigma) in precision["sigma"]!.asArray(Float.self).enumerated() {
                close(QwenImage21Transformer.timeInput(sigma, dtype: dtype),
                    precision[name + "_time"]![index], "\(name) timestep \(index)",
                    tolerance: dtype == .bfloat16 ? 0.008 : dtype == .float16 ? 0.001 : 0.0001)
            }
            let referenceSigmas = try MLX.loadArrays(url: folder.appendingPathComponent("scheduler.safetensors"))["sigmas"]!.asArray(Float.self)
            var sample = precision[name + "_initial"]!
            for step in 0..<40 {
                sample = LocalImageService.eulerStep(sample,
                    prediction: precision[name + "_predictions"]![step], delta: referenceSigmas[step + 1] - referenceSigmas[step])
                precondition(sample.dtype == dtype)
                close(sample, precision[name + "_trajectory"]![step], "\(name) FP32 Euler step \(step)", tolerance: 0.00001)
            }
        }
        let controlWeights = try QwenImage21Weights(directory: folder.appendingPathComponent("control-transformer"))
        let controlTransformer = try QwenImage21Transformer(weights: controlWeights,
            config: LocalImageModelCache.json(folder.appendingPathComponent("control-transformer/config.json")))
        let branch = try QwenImage21Control(weights: QwenImage21Weights(arrays: MLX.loadArrays(url: folder.appendingPathComponent("control.safetensors"))),
            channels: 8, dimension: 32, headDim: 16, blocks: 3)
        let controlInputs = try MLX.loadArrays(url: folder.appendingPathComponent("control-inputs.safetensors"))
        let controlLayout = try QwenImage21Layout(imageSlots: [false, false, false], shapes: [(4, 4)], axes: [4, 6, 6])
        var controlCache: [(MLXArray, MLXArray)] = []
        for (name, scale, t): (String, Float, Float) in [("zero", 0, 0.75), ("half", 0.5, 0.75), ("full", 1, 0.75), ("later", 1, 0.25)] {
            controlCache.removeAll()
            let result = try controlTransformer.predict(latents: controlInputs["latents"]!, text: controlInputs["text"]!,
                timestep: t, layout: controlLayout, cache: &controlCache,
                control: branch, controlContext: controlInputs["context"]!, controlScale: scale)
            close(result, controlInputs[name]!, "Union \(name) skip injection", tolerance: 0.00005)
            if scale > 0 { precondition(controlCache.count == controlTransformer.layers) }
        }
        // Exercise the production API with BOTH prefix caches retained across
        // timesteps (the convenience overload deliberately recomputes them).
        var retainedBase: [(MLXArray, MLXArray)] = []
        var retainedControl: [(MLXArray, MLXArray)] = []
        _ = try controlTransformer.predict(latents: controlInputs["latents"]!, text: controlInputs["text"]!,
            timestep: 0.75, layout: controlLayout, cache: &retainedBase,
            control: branch, controlContext: controlInputs["context"]!, controlScale: 1,
            controlCache: &retainedControl)
        let later = try controlTransformer.predict(latents: controlInputs["latents"]!, text: controlInputs["text"]!,
            timestep: 0.25, layout: controlLayout, cache: &retainedBase,
            control: branch, controlContext: controlInputs["context"]!, controlScale: 1,
            controlCache: &retainedControl)
        close(later, controlInputs["later"]!, "Union cached later timestep", tolerance: 0.00005)
        precondition(retainedControl.count == branch.blocks)
        retainedControl.removeAll()
        do {
            _ = try controlTransformer.predict(latents: controlInputs["latents"]!, text: controlInputs["text"]!,
                timestep: 0.25, layout: controlLayout, cache: &retainedBase,
                control: branch, controlContext: controlInputs["context"]!, controlScale: 1,
                controlCache: &retainedControl)
            preconditionFailure("Missing control prefix must throw")
        } catch LocalImageError.invalid { print("PASS: incomplete control cache rejected") }
        var baseCache: [(MLXArray, MLXArray)] = []
        let baseline = try controlTransformer.predict(latents: controlInputs["latents"]!, text: controlInputs["text"]!,
            timestep: 0.75, layout: controlLayout, cache: &baseCache)
        close(baseline, controlInputs["zero"]!, "Union zero strength equals base")
        precondition(abs(controlInputs["full"]! - controlInputs["zero"]!).max().item(Float.self) > 0.001)
        print("PASS: nonzero Union branches match CPU reference at all injection layers and zero strength restores base")
        print("PASS: native DiT prefix caching, original causal VAE conversion and Euler match CPU Diffusers")
    }

    static func checks(_ service: LocalImageService, cache: SpeechCacheAccess.Configuration, model: LocalImageModel) async throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("synomancer-image-cache-\(UUID())")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        for component in ["transformer", "text_encoder", "vae", "processor", "scheduler"] {
            let source = model.directory.appendingPathComponent(component)
            let target = temporary.appendingPathComponent(component)
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) {
                if file.pathExtension == "safetensors" {
                    try FileManager.default.createSymbolicLink(at: target.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
                } else { try Data(contentsOf: file).write(to: target.appendingPathComponent(file.lastPathComponent)) }
            }
        }
        try Data(contentsOf: model.directory.appendingPathComponent("model_index.json")).write(to: temporary.appendingPathComponent("model_index.json"))
        let complete = try LocalImageModelCache.scan(temporary)
        precondition(complete.count == 1 && complete[0].bytes == model.bytes, "Complete fixture must preserve HF cache contents")
        let shard = try LocalImageModelCache.shards(temporary.appendingPathComponent("vae"))[0]
        let saved = try FileManager.default.destinationOfSymbolicLink(atPath: shard.path)
        try FileManager.default.removeItem(at: shard)
        let missing = try LocalImageModelCache.scan(temporary)
        precondition(missing.isEmpty)
        try FileManager.default.createSymbolicLink(atPath: shard.path, withDestinationPath: "/private/tmp/missing-qwen-image-weight-\(UUID())")
        let broken = try LocalImageModelCache.scan(temporary)
        precondition(broken.isEmpty)
        try FileManager.default.removeItem(at: shard)
        try FileManager.default.createSymbolicLink(atPath: shard.path, withDestinationPath: saved)
        let textIndex = temporary.appendingPathComponent("text_encoder/model.safetensors.index.json")
        let indexData = try Data(contentsOf: textIndex)
        var index = try LocalImageModelCache.json(textIndex)
        let map = index["weight_map"] as! [String: String]
        index["weight_map"] = map.filter { !$0.key.hasPrefix("vision_tower.") }
        try JSONSerialization.data(withJSONObject: index).write(to: textIndex)
        let textOnly = try LocalImageModelCache.scan(temporary)
        precondition(textOnly.isEmpty)
        try indexData.write(to: textIndex)
        print("PASS: missing shards, dangling links and text-only exports are excluded; HF weight bytes are accurate")
        let request = LocalImageRequest(modelID: model.id, prompt: "一只红色陶瓷咖啡杯放在灰色桌面上。",
            width: 512, height: 512, steps: 40, seed: 44)
        let control = ImageCancellationProbe()
        let cancelled = Task {
            try await service.generate(request, cache: cache) { stage, _ in
                if stage.hasPrefix("生成图像") { Task { await control.cancel() } }
            }
        }
        await control.set(cancelled)
        do {
            _ = try await cancelled.value
            preconditionFailure("Cancelled native inference must not produce a completed image")
        } catch is CancellationError {
            print("PASS: cancellation during actual denoising exits with CancellationError")
        }
        // Two steps deliberately test serial-gate recovery, not image quality.
        var restartRequest = request
        restartRequest.steps = 2
        let restarted = try await service.generate(restartRequest, cache: cache) { _, _ in }
        try checkImage(restarted.png)
        print("PASS: native image request starts again immediately after cancellation")
        var differentSeed = restartRequest
        differentSeed.seed += 1
        let cachedStages = Mutex<[String]>([])
        let reused = try await service.generate(differentSeed, cache: cache) { stage, _ in
            cachedStages.withLock { $0.append(stage) }
        }
        precondition(cachedStages.withLock { $0.contains("复用文本和参考图去噪缓存…") })
        await service.clearSessionCache()
        let freshStages = Mutex<[String]>([])
        let fresh = try await service.generate(differentSeed, cache: cache) { stage, _ in
            freshStages.withLock { $0.append(stage) }
        }
        precondition(freshStages.withLock { !$0.contains(where: { $0.hasPrefix("复用") }) })
        try compareImage(reused.png, fresh.png, label: "changed Seed vs fresh inference")
        print("PASS: changed Seed reuses the prefix; clearing the session forces fresh encoding")
        let weights = QwenImage21Weights(arrays: ["test.weight": MLXArray.zeros([2, 3])])
        do {
            _ = try weights.linear(MLXArray.zeros([1, 4]), "test")
            preconditionFailure("Invalid matrix dimensions must throw before launching a kernel")
        } catch LocalImageError.unsupported { print("PASS: incompatible dimensions fail before a Metal kernel") }
    }
}

private actor ImageCancellationProbe {
    private var task: Task<LocalImageResult, any Error>?
    func set(_ task: Task<LocalImageResult, any Error>) { self.task = task }
    func cancel() { task?.cancel() }
}
