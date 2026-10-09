// Native adaptation of Diffusers (Apache-2.0) and MFLUX (MIT).
// Pinned sources and licenses: docs/LocalImages.md.
// Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
// Copyright 2026 The HuggingFace Team. Apache-2.0.
// MFLUX portions: Copyright (c) 2026 Filip Strand. MIT.
import Foundation
import MLX
import MLXNN

final class Flux2KleinTransformer {
    let weights: QwenImage21Weights
    let heads: Int
    let headDim: Int
    let doubleLayers: Int
    let singleLayers: Int
    let axes: [Int]
    let theta: Float
    let timeDim: Int
    let hasGuidance: Bool
    var dim: Int { heads * headDim }

    init(weights: QwenImage21Weights, config: [String: Any]) throws {
        // FP16 Klein context activations can overflow on the full checkpoint.
        // Use the reference pipeline's BF16 compute range, without changing files.
        weights.arrays = weights.arrays.mapValues {
            $0.dtype == .float16 ? $0.asType(.bfloat16) : $0
        }
        self.weights = weights
        guard let heads = config["num_attention_heads"] as? Int,
              let headDim = config["attention_head_dim"] as? Int,
              let doubleLayers = config["num_layers"] as? Int,
              let singleLayers = config["num_single_layers"] as? Int,
              let axes = config["axes_dims_rope"] as? [Int], axes.count == 4,
              axes.allSatisfy({ $0 > 0 && $0 % 2 == 0 }), axes.reduce(0, +) == headDim,
              config["patch_size"] as? Int == 1 else {
            throw LocalImageError.unsupported("FLUX.2 transformer")
        }
        self.heads = heads; self.headDim = headDim
        self.doubleLayers = doubleLayers; self.singleLayers = singleLayers
        self.axes = axes; theta = (config["rope_theta"] as? NSNumber)?.floatValue ?? 2000
        timeDim = config["timestep_guidance_channels"] as? Int ?? 256
        hasGuidance = config["guidance_embeds"] as? Bool ?? true
    }

    private func linear(_ x: MLXArray, _ name: String) throws -> MLXArray { try weights.linear(x, name) }
    private func norm(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
    }
    private func rms(_ x: MLXArray, _ name: String) throws -> MLXArray {
        try MLXFast.rmsNorm(x.asType(.float32), weight: weights.get(name + ".weight").asType(.float32), eps: 1e-5).asType(x.dtype)
    }
    private func headsOf(_ x: MLXArray) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), heads, headDim).transposed(0, 2, 1, 3)
    }
    private func rotary(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        var angles: [MLXArray] = []
        for (axis, size) in axes.enumerated() {
            let frequencies = exp(MLXArray(stride(from: 0, to: size, by: 2).map { -log(theta) * Float($0) / Float(size) }))
            angles.append(ids[0..., axis].asType(.float32).expandedDimensions(axis: -1) * frequencies)
        }
        let angle = concatenated(angles, axis: -1)
        return (cos(angle), sin(angle))
    }
    private func rotate(_ x: MLXArray, _ position: (MLXArray, MLXArray)) -> MLXArray {
        let pairs = x.asType(.float32).reshaped(x.dim(0), heads, x.dim(2), headDim / 2, 2)
        let real = pairs[.ellipsis, 0], imaginary = pairs[.ellipsis, 1]
        return stacked([real * position.0 - imaginary * position.1,
                        imaginary * position.0 + real * position.1], axis: -1).reshaped(x.shape).asType(x.dtype)
    }
    private func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                           _ position: (MLXArray, MLXArray)) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: rotate(q, position), keys: rotate(k, position),
            values: v, scale: 1 / sqrt(Float(headDim)), mask: .none)
            .transposed(0, 2, 1, 3).reshaped(q.dim(0), q.dim(2), dim)
    }
    private func modulated(_ x: MLXArray, _ m: [MLXArray], _ offset: Int) -> MLXArray {
        norm(x) * (1 + m[offset + 1]) + m[offset]
    }
    private func feedForward(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let pair = try split(linear(x, name + ".linear_in"), parts: 2, axis: -1)
        return try linear(silu(pair[0]) * pair[1], name + ".linear_out")
    }

    func callAsFunction(_ latent: MLXArray, text: MLXArray, sigma: Float, ids: MLXArray,
                        guidance: Float = 4) throws -> MLXArray {
        // Preserve the pipeline's dtype rounding before the FP32 sinusoid.
        let phase = ((MLXArray(sigma * 1000).asType(latent.dtype) / 1000).asType(latent.dtype) * 1000).asType(.float32)
        let freq = MLXArray((0..<timeDim / 2).map { exp(-log(Float(10000)) * Float($0) / Float(timeDim / 2)) }) * phase
        let time = concatenated([cos(freq), sin(freq)]).reshaped(1, timeDim).asType(latent.dtype)
        let prefix = "time_guidance_embed.timestep_embedder"
        var temb = try linear(silu(linear(time, prefix + ".linear_1")), prefix + ".linear_2")
        if hasGuidance {
            guard guidance.isFinite && (0...10).contains(guidance) else {
                throw LocalImageError.invalid("FLUX.2-dev Guidance 需在 0–10 之间。")
            }
            let g = (MLXArray(guidance).asType(latent.dtype) * 1000).asType(.float32)
            let gFreq = MLXArray((0..<timeDim / 2).map { exp(-log(Float(10000)) * Float($0) / Float(timeDim / 2)) }) * g
            let embedding = concatenated([cos(gFreq), sin(gFreq)]).reshaped(1, timeDim).asType(latent.dtype)
            let p = "time_guidance_embed.guidance_embedder"
            temb = try temb + linear(silu(linear(embedding, p + ".linear_1")), p + ".linear_2")
        }
        let activated = silu(temb)
        let imageMod = try split(linear(activated, "double_stream_modulation_img.linear").expandedDimensions(axis: 1), parts: 6, axis: -1)
        let textMod = try split(linear(activated, "double_stream_modulation_txt.linear").expandedDimensions(axis: 1), parts: 6, axis: -1)
        let singleMod = try split(linear(activated, "single_stream_modulation.linear").expandedDimensions(axis: 1), parts: 3, axis: -1)
        let position = rotary(ids)
        var image = try linear(latent, "x_embedder")
        var context = try linear(text, "context_embedder")
        let count = text.dim(1)
        for layer in 0..<doubleLayers {
            try Task.checkCancellation()
            let p = "transformer_blocks.\(layer)"
            let a = p + ".attn"
            let x = modulated(image, imageMod, 0), c = modulated(context, textMod, 0)
            let q = try concatenated([rms(headsOf(linear(c, a + ".add_q_proj")), a + ".norm_added_q"),
                                      rms(headsOf(linear(x, a + ".to_q")), a + ".norm_q")], axis: 2)
            let k = try concatenated([rms(headsOf(linear(c, a + ".add_k_proj")), a + ".norm_added_k"),
                                      rms(headsOf(linear(x, a + ".to_k")), a + ".norm_k")], axis: 2)
            let v = try concatenated([headsOf(linear(c, a + ".add_v_proj")), headsOf(linear(x, a + ".to_v"))], axis: 2)
            let attended = attention(q, k, v, position)
            image = try image + imageMod[2] * linear(attended[0..., count..., 0...], a + ".to_out.0")
            context = try context + textMod[2] * linear(attended[0..., ..<count, 0...], a + ".to_add_out")
            image = try image + imageMod[5] * feedForward(modulated(image, imageMod, 3), p + ".ff")
            context = try context + textMod[5] * feedForward(modulated(context, textMod, 3), p + ".ff_context")
            eval(image, context)
        }
        var x = concatenated([context, image], axis: 1)
        for layer in 0..<singleLayers {
            try Task.checkCancellation()
            let p = "single_transformer_blocks.\(layer).attn"
            let projected = try linear(modulated(x, singleMod, 0), p + ".to_qkv_mlp_proj")
            let q = try rms(headsOf(projected[.ellipsis, ..<dim]), p + ".norm_q")
            let k = try rms(headsOf(projected[.ellipsis, dim..<(2 * dim)]), p + ".norm_k")
            let v = headsOf(projected[.ellipsis, (2 * dim)..<(3 * dim)])
            let mlp = split(projected[.ellipsis, (3 * dim)...], parts: 2, axis: -1)
            let merged = concatenated([attention(q, k, v, position), silu(mlp[0]) * mlp[1]], axis: -1)
            x = try x + singleMod[2] * linear(merged, p + ".to_out")
            eval(x)
        }
        let finalMod = try split(linear(activated, "norm_out.linear").expandedDimensions(axis: 1), parts: 2, axis: -1)
        return try linear(norm(x[0..., count..., 0...]) * (1 + finalMod[0]) + finalMod[1], "proj_out")
    }
}
