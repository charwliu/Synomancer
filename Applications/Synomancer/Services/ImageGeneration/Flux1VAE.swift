// AutoencoderKL decoder for FLUX.1, adapted from Diffusers (Apache-2.0).
// Sources and license: docs/LocalImages.md.
// Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
// Copyright 2026 The HuggingFace Team. Apache-2.0.
// Shared convolution implementation: Copyright (c) 2026 Filip Strand. MIT.
import Foundation
import MLX
import MLXNN

final class Flux1VAE {
    let weights: QwenImage21Weights
    let groups: Int
    let blocks: Int
    let layers: Int
    let scale: Float
    let shift: Float

    init(directory: URL) throws {
        weights = try QwenImage21Weights(directory: directory)
        let c = try LocalImageModelCache.json(directory.appendingPathComponent("config.json"))
        guard c["latent_channels"] as? Int == 16,
              c["in_channels"] as? Int == 3,
              c["out_channels"] as? Int == 3,
              c["use_quant_conv"] as? Bool == false,
              c["use_post_quant_conv"] as? Bool == false,
              let channels = c["block_out_channels"] as? [Int], channels.count == 4 else {
            throw LocalImageError.unsupported("FLUX.1 VAE")
        }
        groups = c["norm_num_groups"] as? Int ?? 32
        blocks = channels.count; layers = c["layers_per_block"] as? Int ?? 2
        scale = (c["scaling_factor"] as? NSNumber)?.floatValue ?? 0.3611
        shift = (c["shift_factor"] as? NSNumber)?.floatValue ?? 0.1159
        // Convolution layout is identified by shape, rather than safetensors
        // metadata: community exports may retain Diffusers metadata after conversion.
        let probe = try weights.get("decoder.conv_in.weight")
        let native = probe.shape.count == 4 && probe.dim(1) == 3 && probe.dim(2) == 3 && probe.dim(3) == 16
        weights.arrays = weights.arrays.filter { $0.key.hasPrefix("decoder.") }
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
    func decode(_ packed: MLXArray, height: Int, width: Int) throws -> MLXArray {
        var x = packed.asType(.float32)
            .reshaped(1, height / 16, width / 16, 16, 2, 2).transposed(0, 1, 4, 2, 5, 3)
            .reshaped(1, height / 8, width / 8, 16)
        x = x / scale + shift
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
