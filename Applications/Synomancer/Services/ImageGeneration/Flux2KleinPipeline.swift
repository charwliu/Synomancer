// Shared native MLX FLUX.2 Dev/Klein pipeline. See docs/LocalImages.md.
// Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
// Copyright 2026 The HuggingFace Team. Apache-2.0.
// MFLUX portions: Copyright (c) 2026 Filip Strand. MIT.
import Foundation
@preconcurrency import CoreImage
import MLX

/// Invoked only by LocalImageService while its GPU serialization lock is held.
enum Flux2KleinPipeline {
    static func sigmas(steps: Int, tokens: Int) -> [Float] {
        let n = Double(tokens)
        let m200 = 0.00016927 * n + 0.45666666
        let m10 = 8.73809524e-05 * n + 1.89833333
        let slope = (m200 - m10) / 190
        let mu = tokens > 4300 ? m200 : slope * Double(steps) + m200 - 200 * slope
        let shift = exp(mu)
        return (0..<steps).map { i in
            let t = 1 - Double(i) / Double(steps)
            return Float(shift / (shift + 1 / t - 1))
        } + [0]
    }

    static func imageIDs(height: Int, width: Int, time: Int = 0) -> MLXArray {
        MLXArray((0..<height).flatMap { y in (0..<width).flatMap { x in [Int32(time), Int32(y), Int32(x), 0] } })
            .reshaped(height * width, 4)
    }

    static func referenceSize(width: Int, height: Int) throws -> (Int, Int) {
        guard width > 0, height > 0 else { throw LocalImageError.invalid("参考图尺寸无效。") }
        let scale = min(1, sqrt(1_048_576 / (Double(width) * Double(height))))
        let w = Int((Double(width) * scale).rounded()) / 16 * 16
        let h = Int((Double(height) * scale).rounded()) / 16 * 16
        guard w >= 16, h >= 16 else { throw LocalImageError.invalid("Klein 参考图的宽和高至少需要 16 像素。") }
        return (w, h)
    }

    static func reference(_ data: Data) throws -> MLXArray {
        guard let source = CIImage(data: data, options: [.applyOrientationProperty: true]) else {
            throw LocalImageError.invalid("无法读取参考照片。")
        }
        let (width, height) = try referenceSize(width: Int(source.extent.width), height: Int(source.extent.height))
        let scale = min(1, sqrt(1_048_576 / (source.extent.width * source.extent.height)))
        var image = source.transformed(by: .init(translationX: -source.extent.minX, y: -source.extent.minY))
            .transformed(by: .init(scaleX: scale, y: scale))
        image = image.transformed(by: .init(translationX: -(image.extent.width - CGFloat(width)) / 2,
                                             y: -(image.extent.height - CGFloat(height)) / 2))
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        image = image.composited(over: CIImage(color: .white).cropped(to: rect)).cropped(to: rect)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        CIContext(options: [.cacheIntermediates: false]).render(image, toBitmap: &pixels, rowBytes: width * 4,
            bounds: rect, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        return MLXArray(pixels).reshaped(1, height, width, 4)[.ellipsis, ..<3].asType(.float32) / 127.5 - 1
    }

    static func generate(_ request: LocalImageRequest, model: LocalImageModel,
                         progress: @escaping @Sendable (String, Double) -> Void) async throws -> LocalImageResult {
        try request.validateNegativePrompt(for: model.family)
        guard request.control == nil else {
            throw LocalImageError.invalid("\(model.family.rawValue) 支持参考图编辑，暂不支持 Qwen ControlNet；请清除结构控制图。")
        }
        let dev = model.family == .flux2Dev
        let title = dev ? "FLUX.2-dev" : "Klein"
        let start = ContinuousClock.now
        progress("编码 \(title) 提示词…", 0.01)
        let encodedTokens: ([Int], Int)
        if dev {
            encodedTokens = try await Flux2DevTextEncoder.tokens(request.prompt,
                directory: model.directory.appendingPathComponent("tokenizer"))
        } else {
            encodedTokens = try await Flux2KleinTextEncoder.tokens(request.prompt,
                directory: model.directory.appendingPathComponent("tokenizer"))
        }
        let (tokens, validCount) = encodedTokens
        try Task.checkCancellation()
        let text: MLXArray
        do {
            if dev {
                let encoder = try Flux2DevTextEncoder(directory: model.directory.appendingPathComponent("text_encoder"))
                text = try encoder.encode(tokens, validCount: validCount)
            } else {
                let encoder = try Flux2KleinTextEncoder(directory: model.directory.appendingPathComponent("text_encoder"))
                text = try encoder.encode(tokens, validCount: validCount)
            }
            eval(text)
        }
        Memory.clearCache()
        try Task.checkCancellation()
        progress("加载 \(title) VAE…", 0.12)
        let vae = try Flux2KleinVAE(directory: model.directory.appendingPathComponent("vae"))
        var referenceLatent: MLXArray?
        var referenceIDs: MLXArray?
        if let data = request.reference {
            progress("编码参考图…", 0.15)
            let rgb = try reference(data)
            referenceLatent = try vae.encode(rgb).asType(text.dtype)
            eval(referenceLatent!)
            referenceIDs = imageIDs(height: rgb.dim(1) / 16, width: rgb.dim(2) / 16, time: 10)
        }
        progress("加载 \(model.family.rawValue)…", 0.20)
        if dev { try Flux2DevPipeline.checkMemory(model) }
        let folder = model.directory.appendingPathComponent("transformer")
        let transformer = try Flux2KleinTransformer(weights: QwenImage21Weights(directory: folder),
            config: LocalImageModelCache.json(folder.appendingPathComponent("config.json")))
        let h = request.height / 16, w = request.width / 16
        // Match MLX's NCHW noise ordering before packing spatial tokens.
        var latent = MLXRandom.normal([1, 128, h, w], key: MLXRandom.key(request.seed)).asType(text.dtype)
            .reshaped(1, 128, h * w).transposed(0, 2, 1)
        let textIDs = MLXArray((0..<512).flatMap { [Int32(0), 0, 0, Int32($0)] }).reshaped(512, 4)
        var ids = concatenated([textIDs, imageIDs(height: h, width: w)], axis: 0)
        if let referenceIDs { ids = concatenated([ids, referenceIDs], axis: 0) }
        let schedule = sigmas(steps: request.steps, tokens: h * w)
        for step in 0..<request.steps {
            try Task.checkCancellation()
            progress("\(title) 去噪 \(step + 1)/\(request.steps)…", 0.22 + 0.65 * Double(step) / Double(request.steps))
            let input = referenceLatent.map { concatenated([latent, $0], axis: 1) } ?? latent
            let noise = try transformer(input, text: text, sigma: schedule[step], ids: ids,
                guidance: request.guidance)[0..., ..<(h * w), 0...]
            latent = LocalImageService.eulerStep(latent, prediction: noise, delta: schedule[step + 1] - schedule[step])
            eval(latent)
        }
        try Task.checkCancellation()
        progress("解码 \(title) 图像…", 0.9)
        let rgb = try vae.decode(latent, height: request.height, width: request.width)
        eval(rgb)
        guard all(isFinite(rgb)).item(Bool.self) else { throw LocalImageError.nonfinite }
        let rgba = concatenated([rgb, MLXArray.ones([1, request.height, request.width, 1])], axis: -1)
        let png = try LocalImageService.png(rgba)
        try Task.checkCancellation()
        let elapsed = start.duration(to: .now)
        progress("完成", 1)
        return .init(png: png, seed: request.seed,
            seconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18, modelID: model.id)
    }
}
