// Native Swift/MLX adaptation of Hugging Face Diffusers FluxTransformer2DModel.
// Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
// Apache-2.0. Sources and license: docs/LocalImages.md.
import Foundation
import MLX
import MLXNN

final class Flux1Transformer {
    let weights: QwenImage21Weights
    let heads: Int, headDim: Int, doubleLayers: Int, singleLayers: Int
    let axes: [Int]
    var dim: Int { heads * headDim }

    init(weights: QwenImage21Weights, config: [String: Any]) throws {
        guard let heads = config["num_attention_heads"] as? Int,
              let headDim = config["attention_head_dim"] as? Int,
              let doubleLayers = config["num_layers"] as? Int,
              let singleLayers = config["num_single_layers"] as? Int,
              config["guidance_embeds"] as? Bool == true,
              config["patch_size"] as? Int == 1 else {
            throw LocalImageError.unsupported("FLUX.1-dev transformer")
        }
        let axes = config["axes_dims_rope"] as? [Int] ?? [16, 56, 56]
        guard axes.count == 3, axes.allSatisfy({ $0 > 0 && $0 % 2 == 0 }),
              axes.reduce(0, +) == headDim else { throw LocalImageError.unsupported("FLUX.1 RoPE") }
        self.weights = weights; self.heads = heads; self.headDim = headDim
        self.doubleLayers = doubleLayers; self.singleLayers = singleLayers; self.axes = axes
    }

    private func linear(_ x: MLXArray, _ name: String) throws -> MLXArray { try weights.linear(x, name) }
    private func norm(_ x: MLXArray) -> MLXArray { MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6) }
    private func rms(_ x: MLXArray, _ name: String) throws -> MLXArray {
        try MLXFast.rmsNorm(x.asType(.float32), weight: weights.get(name + ".weight").asType(.float32), eps: 1e-6).asType(x.dtype)
    }
    private func headsOf(_ x: MLXArray) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), heads, headDim).transposed(0, 2, 1, 3)
    }
    private func rotary(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        let angles = axes.enumerated().map { axis, size in
            let frequencies = MLXArray(stride(from: 0, to: size, by: 2).map {
                exp(-log(Float(10000)) * Float($0) / Float(size))
            })
            return ids[0..., axis].asType(.float32).expandedDimensions(axis: -1) * frequencies
        }
        let angle = concatenated(angles, axis: -1)
        return (cos(angle), sin(angle))
    }
    private func rotate(_ x: MLXArray, _ position: (MLXArray, MLXArray)) -> MLXArray {
        let pairs = x.asType(.float32).reshaped(x.dim(0), heads, x.dim(2), headDim / 2, 2)
        let a = pairs[.ellipsis, 0], b = pairs[.ellipsis, 1]
        return stacked([a * position.0 - b * position.1, b * position.0 + a * position.1], axis: -1)
            .reshaped(x.shape).asType(x.dtype)
    }
    private func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                           _ position: (MLXArray, MLXArray)) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: rotate(q, position), keys: rotate(k, position),
            values: v, scale: 1 / sqrt(Float(headDim)), mask: .none)
            .transposed(0, 2, 1, 3).reshaped(q.dim(0), q.dim(2), dim)
    }
    private func modulated(_ x: MLXArray, _ m: [MLXArray], _ offset: Int = 0) -> MLXArray {
        norm(x) * (1 + m[offset + 1]) + m[offset]
    }
    private func ff(_ x: MLXArray, _ prefix: String) throws -> MLXArray {
        try linear(geluApproximate(linear(x, prefix + ".net.0.proj")), prefix + ".net.2")
    }
    private func embedding(_ value: Float, dtype: DType, name: String) throws -> MLXArray {
        let phase = MLXArray(value).asType(dtype).asType(.float32)
        let frequency = MLXArray((0..<128).map { exp(-log(Float(10000)) * Float($0) / 128) }) * phase
        let sinusoid = concatenated([cos(frequency), sin(frequency)]).reshaped(1, 256).asType(dtype)
        return try linear(silu(linear(sinusoid, name + ".linear_1")), name + ".linear_2")
    }

    func callAsFunction(_ latent: MLXArray, text: MLXArray, pooled: MLXArray,
                        sigma: Float, guidance: Float, ids: MLXArray) throws -> MLXArray {
        let time = "time_text_embed"
        let roundedTime = ((MLXArray(sigma * 1000).asType(latent.dtype) / 1000).asType(latent.dtype) * 1000).item(Float.self)
        let temb = try embedding(roundedTime, dtype: latent.dtype, name: time + ".timestep_embedder")
            + embedding(guidance * 1000, dtype: latent.dtype, name: time + ".guidance_embedder")
            + linear(silu(linear(pooled, time + ".text_embedder.linear_1")), time + ".text_embedder.linear_2")
        let activated = silu(temb)
        let position = rotary(ids)
        var image = try linear(latent, "x_embedder"), context = try linear(text, "context_embedder")
        let count = text.dim(1)
        for layer in 0..<doubleLayers {
            try Task.checkCancellation()
            let p = "transformer_blocks.\(layer)", a = p + ".attn"
            let im = try split(linear(activated, p + ".norm1.linear").expandedDimensions(axis: 1), parts: 6, axis: -1)
            let tm = try split(linear(activated, p + ".norm1_context.linear").expandedDimensions(axis: 1), parts: 6, axis: -1)
            let x = modulated(image, im), c = modulated(context, tm)
            let q = try concatenated([rms(headsOf(linear(c, a + ".add_q_proj")), a + ".norm_added_q"),
                                      rms(headsOf(linear(x, a + ".to_q")), a + ".norm_q")], axis: 2)
            let k = try concatenated([rms(headsOf(linear(c, a + ".add_k_proj")), a + ".norm_added_k"),
                                      rms(headsOf(linear(x, a + ".to_k")), a + ".norm_k")], axis: 2)
            let v = try concatenated([headsOf(linear(c, a + ".add_v_proj")), headsOf(linear(x, a + ".to_v"))], axis: 2)
            let attended = attention(q, k, v, position)
            image = try image + im[2] * linear(attended[0..., count..., 0...], a + ".to_out.0")
            context = try context + tm[2] * linear(attended[0..., ..<count, 0...], a + ".to_add_out")
            image = try image + im[5] * ff(modulated(image, im, 3), p + ".ff")
            context = try context + tm[5] * ff(modulated(context, tm, 3), p + ".ff_context")
            eval(image, context)
        }
        var x = concatenated([context, image], axis: 1)
        for layer in 0..<singleLayers {
            try Task.checkCancellation()
            let p = "single_transformer_blocks.\(layer)", a = p + ".attn"
            let m = try split(linear(activated, p + ".norm.linear").expandedDimensions(axis: 1), parts: 3, axis: -1)
            let n = modulated(x, m)
            let q = try rms(headsOf(linear(n, a + ".to_q")), a + ".norm_q")
            let k = try rms(headsOf(linear(n, a + ".to_k")), a + ".norm_k")
            let v = try headsOf(linear(n, a + ".to_v"))
            let merged = try concatenated([attention(q, k, v, position), geluApproximate(linear(n, p + ".proj_mlp"))], axis: -1)
            x = try x + m[2] * linear(merged, p + ".proj_out")
            eval(x)
        }
        let final = try split(linear(activated, "norm_out.linear").expandedDimensions(axis: 1), parts: 2, axis: -1)
        return try linear(norm(x[0..., count..., 0...]) * (1 + final[0]) + final[1], "proj_out")
    }
}
