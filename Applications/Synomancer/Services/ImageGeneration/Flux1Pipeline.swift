// Native Swift/MLX adaptation of Diffusers FluxPipeline (Apache-2.0).
// Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
// Sources and license: docs/LocalImages.md.
import Foundation
import MLX
import Tokenizers

/// Runs under LocalImageService's serialization gate. No network or Python runtime.
enum Flux1Pipeline {
    static func sigmas(steps: Int, tokens: Int, config: [String: Any]) throws -> [Float] {
        guard steps > 0, config["_class_name"] as? String == "FlowMatchEulerDiscreteScheduler",
              config["use_dynamic_shifting"] as? Bool == true else {
            throw LocalImageError.unsupported("FLUX.1 dynamic Euler schedule")
        }
        let baseLength = (config["base_image_seq_len"] as? NSNumber)?.doubleValue ?? 256
        let maxLength = (config["max_image_seq_len"] as? NSNumber)?.doubleValue ?? 4096
        let base = (config["base_shift"] as? NSNumber)?.doubleValue ?? 0.5
        let maxShift = (config["max_shift"] as? NSNumber)?.doubleValue ?? 1.15
        guard maxLength > baseLength else { throw LocalImageError.unsupported("FLUX.1 schedule lengths") }
        let mu = base + (Double(tokens) - baseLength) * (maxShift - base) / (maxLength - baseLength)
        let shift = exp(mu)
        // Diffusers starts with linspace(1, 1/steps, steps) and appends zero.
        return (0..<steps).map { i in
            let t = 1 - Double(i) / Double(steps)
            return Float(shift / (shift + 1 / t - 1))
        } + [0]
    }
    static func imageIDs(height: Int, width: Int) -> MLXArray {
        MLXArray((0..<height).flatMap { y in (0..<width).flatMap { x in [Int32(0), Int32(y), Int32(x)] } })
            .reshaped(height * width, 3)
    }
    static func validate(_ request: LocalImageRequest) throws {
        try request.validateNegativePrompt(for: .flux1Dev)
        guard request.reference == nil, request.control == nil else {
            throw LocalImageError.invalid("FLUX.1-dev 当前仅支持文生图。请移除参考照片和控制图，或切换到支持编辑的模型。")
        }
        guard request.guidance.isFinite, request.guidance >= 0, request.guidance <= 10 else {
            throw LocalImageError.invalid("FLUX guidance 必须在 0–10 之间。")
        }
    }
    static func generate(_ request: LocalImageRequest, model: LocalImageModel,
                         progress: @escaping @Sendable (String, Double) -> Void) async throws -> LocalImageResult {
        try validate(request)
        let start = ContinuousClock.now
        progress("编码 FLUX CLIP 提示词…", 0.01)
        let clipTokenizer = try Flux1CLIPTokenizer(directory: model.directory.appendingPathComponent("tokenizer"))
        let (clipTokens, pooledIndex) = try clipTokenizer.encode(request.prompt)
        let pooled: MLXArray
        var negativePooled: MLXArray?
        do {
            let encoder = try Flux1TextEncoder(directory: model.directory.appendingPathComponent("text_encoder"))
            pooled = try encoder.clip(clipTokens, pooledIndex: pooledIndex)
            eval(pooled)
            if let negative = request.activeNegativePrompt {
                let (ids, index) = try clipTokenizer.encode(negative)
                negativePooled = try encoder.clip(ids, pooledIndex: index)
                eval(negativePooled!)
            }
        }
        Memory.clearCache()
        try Task.checkCancellation()
        progress("编码 FLUX T5 提示词…", 0.05)
        let tokenizer = try await AutoTokenizer.from(modelFolder: model.directory.appendingPathComponent("tokenizer_2"))
        let tokens = try Flux1TextEncoder.t5Tokens(request.prompt, tokenizer: tokenizer)
        let text: MLXArray
        var negativeText: MLXArray?
        do {
            let encoder = try Flux1TextEncoder(directory: model.directory.appendingPathComponent("text_encoder_2"))
            text = try encoder.t5(tokens)
            eval(text)
            if let negative = request.activeNegativePrompt {
                progress("编码 FLUX 负面提示词…", 0.08)
                negativeText = try encoder.t5(Flux1TextEncoder.t5Tokens(negative, tokenizer: tokenizer))
                eval(negativeText!)
            }
        }
        Memory.clearCache()
        try Task.checkCancellation()
        progress("加载 FLUX.1-dev…", 0.12)
        let folder = model.directory.appendingPathComponent("transformer")
        let transformer = try Flux1Transformer(weights: QwenImage21Weights(directory: folder),
            config: LocalImageModelCache.json(folder.appendingPathComponent("config.json")))
        let h = request.height / 16, w = request.width / 16
        // Pack NCHW 16-channel noise as (channel, patch-y, patch-x).
        var latent = MLXRandom.normal([1, 16, h * 2, w * 2], key: MLXRandom.key(request.seed)).asType(text.dtype)
            .reshaped(1, 16, h, 2, w, 2).transposed(0, 2, 4, 1, 3, 5).reshaped(1, h * w, 64)
        let ids = concatenated([MLXArray.zeros([tokens.count, 3], dtype: .int32), imageIDs(height: h, width: w)], axis: 0)
        let negativeIDs = negativeText.map {
            concatenated([MLXArray.zeros([$0.dim(1), 3], dtype: .int32), imageIDs(height: h, width: w)], axis: 0)
        }
        let schedule = try sigmas(steps: request.steps, tokens: h * w,
            config: LocalImageModelCache.json(model.directory.appendingPathComponent("scheduler/scheduler_config.json")))
        for step in 0..<request.steps {
            try Task.checkCancellation()
            progress("FLUX.1 去噪 \(step + 1)/\(request.steps)…", 0.2 + 0.65 * Double(step) / Double(request.steps))
            var prediction = try transformer(latent, text: text, pooled: pooled,
                sigma: schedule[step], guidance: request.guidance, ids: ids)
            if let negativeText, let negativePooled, let negativeIDs {
                eval(prediction)
                try Task.checkCancellation()
                let negative = try transformer(latent, text: negativeText, pooled: negativePooled,
                    sigma: schedule[step], guidance: request.guidance, ids: negativeIDs)
                prediction = LocalImageService.classifierFreeGuidance(positive: prediction, negative: negative, scale: request.trueCFGScale)
            }
            latent = LocalImageService.eulerStep(latent, prediction: prediction, delta: schedule[step + 1] - schedule[step])
            eval(latent)
        }
        try Task.checkCancellation()
        progress("解码 FLUX 图像…", 0.9)
        let vae = try Flux1VAE(directory: model.directory.appendingPathComponent("vae"))
        let rgb = try vae.decode(latent, height: request.height, width: request.width)
        eval(rgb)
        guard all(isFinite(rgb)).item(Bool.self) else { throw LocalImageError.nonfinite }
        let rgba = concatenated([rgb, MLXArray.ones([1, request.height, request.width, 1])], axis: -1)
        let png = try LocalImageService.png(rgba)
        try Task.checkCancellation()
        let elapsed = start.duration(to: .now).components
        progress("完成", 1)
        return .init(png: png, seed: request.seed,
            seconds: Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18, modelID: model.id)
    }
}
