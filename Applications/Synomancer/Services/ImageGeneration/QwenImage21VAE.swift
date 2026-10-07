// Copyright 2026 The Qwen Team and The HuggingFace Team. Apache-2.0.
// Single-frame residual RGBA VAE adapted from Diffusers/MFLUX; see docs/LocalImages.md.
import Foundation
import MLX
import MLXNN

final class QwenImage21VAE {
    let weights: QwenImage21Weights
    let meanValue: MLXArray, stdValue: MLXArray
    let dims: [Int], decoderDims: [Int], temporal: [Bool], blocks: Int, channels: Int

    init(weights: QwenImage21Weights, config: [String: Any]) throws {
        self.weights = weights
        guard let multipliers = config["dim_mult"] as? [Int], multipliers.count == 5,
              let base = config["base_dim"] as? Int, let decoderBase = config["decoder_base_dim"] as? Int,
              let temporal = (config["temporal_downsample"] ?? config["temperal_downsample"]) as? [Bool],
              let channels = config["z_dim"] as? Int,
              let mean = (config["latents_mean"] as? [NSNumber])?.map(\.floatValue),
              let std = (config["latents_std"] as? [NSNumber])?.map(\.floatValue),
              mean.count == channels, std.count == channels, std.allSatisfy({ $0 > 0 }),
              config["is_residual"] as? Bool == true,
              config["in_channels"] as? Int == 4, config["out_channels"] as? Int == 4 else {
            throw LocalImageError.unsupported("RGBA residual VAE")
        }
        self.channels = channels; self.temporal = temporal
        blocks = config["num_res_blocks"] as? Int ?? 2
        dims = ([1] + multipliers).map { base * $0 }
        decoderDims = ([multipliers.last!] + Array(multipliers.reversed())).map { decoderBase * $0 }
        meanValue = MLXArray(mean); stdValue = MLXArray(std)
        guard !weights.arrays.keys.contains(where: { $0.hasSuffix(".scales") }) else {
            throw LocalImageError.unsupported("quantized VAE; use an unquantized VAE component")
        }
        // Original causal 3D convolutions reduce exactly to their last temporal
        // slice for a single frame. MLX exports already use OHWI convolution weights.
        for (key, original) in weights.arrays {
            let value = original.asType(.float32)
            weights.arrays[key] = value
            if key.hasSuffix(".gamma") { weights.arrays[key] = value.reshaped(-1) }
            else if key.hasSuffix(".weight") && !weights.native {
                if value.ndim == 5 { weights.arrays[key] = value[0..., 0..., value.dim(2) - 1, 0..., 0...].transposed(0, 2, 3, 1) }
                else if value.ndim == 4 { weights.arrays[key] = value.transposed(0, 2, 3, 1) }
            }
        }
    }

    private func conv(_ x: MLXArray, _ name: String, stride: Int = 1, padding: Int = 1) throws -> MLXArray {
        let weight = try weights.get(name + ".weight")
        guard weight.ndim == 4, weight.dim(-1) == x.dim(-1) else { throw LocalImageError.unsupported(name + " convolution shape") }
        return MLX.conv2d(x, weight, stride: .init(stride), padding: .init(padding)) + (try weights.get(name + ".bias"))
    }

    private func norm(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let value = x.asType(.float32)
        let magnitude = maximum(sqrt(sum(value * value, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return (value / magnitude).asType(x.dtype) * sqrt(Float(x.dim(-1))) * (try weights.get(name + ".gamma"))
    }

    private func residual(_ input: MLXArray, _ name: String) throws -> MLXArray {
        let shortcut = weights.arrays[name + ".conv_shortcut.weight"] == nil ? input : try conv(input, name + ".conv_shortcut", padding: 0)
        let x = try conv(silu(norm(input, name + ".norm1")), name + ".conv1")
        return shortcut + (try conv(silu(norm(x, name + ".norm2")), name + ".conv2"))
    }

    private func middle(_ input: MLXArray, _ name: String) throws -> MLXArray {
        var x = try residual(input, name + ".resnets.0")
        let shape = x.shape, channels = x.dim(-1)
        let packed = try conv(norm(x, name + ".attentions.0.norm"), name + ".attentions.0.to_qkv", padding: 0)
            .reshaped(shape[0], 1, shape[1] * shape[2], 3 * channels).split(parts: 3, axis: -1)
        let output = MLXFast.scaledDotProductAttention(queries: packed[0], keys: packed[1], values: packed[2], scale: 1 / sqrt(Float(channels)), mask: .none)
        x = x + (try conv(output.reshaped(shape), name + ".attentions.0.proj", padding: 0))
        return try residual(x, name + ".resnets.1")
    }

    private func averageShortcut(_ input: MLXArray, outChannels: Int, factorT: Int, spatial: Int) -> MLXArray {
        let x = input.transposed(0, 3, 1, 2)
        let batch = x.dim(0), channels = x.dim(1), height = x.dim(2), width = x.dim(3)
        var temporalInput = x[0..., 0..., .newAxis, 0..., 0...]
        if factorT == 2 { temporalInput = padded(temporalInput, widths: [.init(0), .init(0), .init((1, 0)), .init(0), .init(0)]) }
        let factor = factorT * spatial * spatial
        let rearranged = temporalInput.reshaped(batch, channels, 1, factorT, height / spatial, spatial, width / spatial, spatial)
            .transposed(0, 1, 3, 5, 7, 2, 4, 6).reshaped(batch, outChannels, channels * factor / outChannels, height / spatial, width / spatial)
        return MLX.mean(rearranged, axis: 2).transposed(0, 2, 3, 1)
    }

    private func duplicateShortcut(_ input: MLXArray, outChannels: Int, factorT: Int) -> MLXArray {
        let x = input.transposed(0, 3, 1, 2)
        let batch = x.dim(0), channels = x.dim(1), height = x.dim(2), width = x.dim(3)
        let expanded = repeated(x, count: outChannels * factorT * 4 / channels, axis: 1)
            .reshaped(batch, outChannels, factorT, 2, 2, 1, height, width).transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped(batch, outChannels, factorT, height * 2, width * 2)
        return expanded[0..., 0..., factorT - 1, 0..., 0...].transposed(0, 2, 3, 1)
    }

    /// NHWC posterior mean, matching reference-image conditioning.
    func encode(_ rgba: MLXArray) throws -> MLXArray {
        var x = try conv(rgba.asType(.float32), "encoder.conv_in")
        for block in 0..<(dims.count - 1) {
            try Task.checkCancellation()
            let down = block < dims.count - 2
            let factorT = block < temporal.count && temporal[block] ? 2 : 1
            let shortcut = averageShortcut(x, outChannels: dims[block + 1], factorT: factorT, spatial: down ? 2 : 1)
            let name = "encoder.down_blocks.\(block)"
            for resnet in 0..<blocks { x = try residual(x, name + ".resnets.\(resnet)") }
            if down {
                x = padded(x, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
                x = try conv(x, name + ".downsampler.resample.1", stride: 2, padding: 0)
            }
            x = x + shortcut; eval(x)
        }
        x = try middle(x, "encoder.mid_block")
        x = try conv(silu(norm(x, "encoder.norm_out")), "encoder.conv_out")
        let latent = try conv(x, "quant_conv", padding: 0)[0..., 0..., 0..., ..<channels]
        return (latent - meanValue) / stdValue
    }

    func decode(_ latent: MLXArray) throws -> MLXArray {
        var x = try conv(latent.asType(.float32) * stdValue + meanValue, "post_quant_conv", padding: 0)
        x = try middle(conv(x, "decoder.conv_in"), "decoder.mid_block")
        let reverseTemporal = Array(temporal.reversed())
        for block in 0..<(decoderDims.count - 1) {
            try Task.checkCancellation()
            let original = x, name = "decoder.up_blocks.\(block)"
            for resnet in 0...blocks { x = try residual(x, name + ".resnets.\(resnet)") }
            if block < decoderDims.count - 2 {
                let factorT = block < reverseTemporal.count && reverseTemporal[block] ? 2 : 1
                let shortcut = duplicateShortcut(original, outChannels: decoderDims[block + 1], factorT: factorT)
                x = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                x = try conv(x, name + ".upsampler.resample.1") + shortcut
            }
            eval(x)
        }
        return try clip(conv(silu(norm(x, "decoder.norm_out")), "decoder.conv_out"), min: -1, max: 1)
    }
}
