// AutoencoderKLFlux2, adapted from Diffusers and MFLUX; see docs/LocalImages.md.
// Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
// Copyright 2026 The HuggingFace Team. Apache-2.0.
// MFLUX portions: Copyright (c) 2026 Filip Strand. MIT.
import Foundation
import MLX
import MLXNN

final class Flux2KleinVAE {
    let weights: QwenImage21Weights
    let groups: Int
    let blocks: Int
    let layers: Int
    let batchNormEps: Float

    init(directory: URL) throws {
        weights = try QwenImage21Weights(directory: directory)
        let c = try LocalImageModelCache.json(directory.appendingPathComponent("config.json"))
        guard c["latent_channels"] as? Int == 32,
              c["in_channels"] as? Int == 3,
              c["out_channels"] as? Int == 3,
              c["use_quant_conv"] as? Bool == true,
              c["use_post_quant_conv"] as? Bool == true,
              let channels = c["block_out_channels"] as? [Int], channels.count == 4 else {
            throw LocalImageError.unsupported("FLUX.2 VAE")
        }
        groups = c["norm_num_groups"] as? Int ?? 32
        blocks = channels.count; layers = c["layers_per_block"] as? Int ?? 2
        batchNormEps = (c["batch_norm_eps"] as? NSNumber)?.floatValue ?? 1e-4
        // Convolution layout is identified by shape, rather than safetensors
        // metadata: community exports may retain Diffusers metadata after conversion.
        let probe = try weights.get("decoder.conv_in.weight")
        let native = probe.shape.count == 4 && probe.dim(1) == 3 && probe.dim(2) == 3 && probe.dim(3) == 32
        for (name, value) in weights.arrays {
            if value.ndim == 4 && name.hasSuffix(".weight") {
                weights.arrays[name] = (native ? value : value.transposed(0, 2, 3, 1)).asType(.float32)
            } else { weights.arrays[name] = value.asType(.float32) }
        }
    }

    private func conv(_ x: MLXArray, _ name: String, stride: Int = 1, padding: Int = 1) throws -> MLXArray {
        try MLX.conv2d(x, weights.get(name + ".weight"), stride: .init(stride), padding: .init(padding)) + weights.get(name + ".bias")
    }
    private func norm(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let n = x.transposed(0, 3, 1, 2).reshaped(x.dim(0), groups, -1)
        let mean = n.mean(axis: -1, keepDims: true)
        let centered = n - mean
        let value = (centered * rsqrt((centered * centered).mean(axis: -1, keepDims: true) + 1e-6))
            .reshaped(x.dim(0), x.dim(3), x.dim(1), x.dim(2)).transposed(0, 2, 3, 1)
        return try value * weights.get(name + ".weight") + weights.get(name + ".bias")
    }
    private func residual(_ x: MLXArray, _ name: String) throws -> MLXArray {
        var y = try conv(silu(norm(x, name + ".norm1")), name + ".conv1")
        y = try conv(silu(norm(y, name + ".norm2")), name + ".conv2")
        let skip = try weights.arrays[name + ".conv_shortcut.weight"] == nil ? x : conv(x, name + ".conv_shortcut", padding: 0)
        let result = y + skip
        eval(result)
        return result
    }
    private func mid(_ x: MLXArray, _ name: String) throws -> MLXArray {
        var y = try residual(x, name + ".resnets.0")
        let a = name + ".attentions.0"
        let n = try norm(y, a + ".group_norm").reshaped(1, -1, y.dim(3))
        let q = try weights.linear(n, a + ".to_q").expandedDimensions(axis: 1)
        let k = try weights.linear(n, a + ".to_k").expandedDimensions(axis: 1)
        let v = try weights.linear(n, a + ".to_v").expandedDimensions(axis: 1)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(y.dim(3))), mask: .none)
            .reshaped(1, -1, y.dim(3))
        y = try y + weights.linear(attended, a + ".to_out.0").reshaped(y.shape)
        return try residual(y, name + ".resnets.1")
    }
    func encode(_ rgb: MLXArray) throws -> MLXArray {
        var x = try conv(rgb.asType(.float32), "encoder.conv_in")
        for block in 0..<blocks {
            try Task.checkCancellation()
            let p = "encoder.down_blocks.\(block)"
            for layer in 0..<layers { x = try residual(x, p + ".resnets.\(layer)") }
            if block < blocks - 1 {
                x = padded(x, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
                x = try conv(x, p + ".downsamplers.0.conv", stride: 2, padding: 0)
            }
        }
        x = try mid(x, "encoder.mid_block")
        x = try conv(silu(norm(x, "encoder.conv_norm_out")), "encoder.conv_out")
        x = try conv(x, "quant_conv", padding: 0)[.ellipsis, ..<32]
        // Channel order: c, patch-y, patch-x, as in the trained VAE's BN.
        let h = x.dim(1) / 2, w = x.dim(2) / 2
        x = x.reshaped(1, h, 2, w, 2, 32).transposed(0, 1, 3, 5, 2, 4).reshaped(1, h * w, 128)
        return try (x - weights.get("bn.running_mean")) * rsqrt(weights.get("bn.running_var") + batchNormEps)
    }
    func decode(_ packed: MLXArray, height: Int, width: Int) throws -> MLXArray {
        var x = try packed.asType(.float32) * sqrt(weights.get("bn.running_var") + batchNormEps) + weights.get("bn.running_mean")
        x = x.reshaped(1, height / 16, width / 16, 32, 2, 2).transposed(0, 1, 4, 2, 5, 3)
            .reshaped(1, height / 8, width / 8, 32)
        x = try conv(x, "post_quant_conv", padding: 0)
        x = try conv(x, "decoder.conv_in")
        x = try mid(x, "decoder.mid_block")
        for block in 0..<blocks {
            try Task.checkCancellation()
            let p = "decoder.up_blocks.\(block)"
            for layer in 0...layers { x = try residual(x, p + ".resnets.\(layer)") }
            if block < blocks - 1 {
                x = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                x = try conv(x, p + ".upsamplers.0.conv")
            }
        }
        return try conv(silu(norm(x, "decoder.conv_norm_out")), "decoder.conv_out")
    }
}
