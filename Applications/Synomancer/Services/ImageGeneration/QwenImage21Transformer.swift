// Copyright 2026 The Qwen Team and The HuggingFace Team. Apache-2.0.
// Swift/MLX adaptation of the block-causal reference path in MFLUX (MIT).
// Sources, pinned revisions and licenses: docs/LocalImages.md.
import Foundation
import MLX
import MLXNN

struct QwenImage21Layout {
    let repeatIndices: MLXArray
    let imageIndices: MLXArray
    let targetMask: MLXArray
    let cos: MLXArray
    let sin: MLXArray
    let segments: [(start: Int, end: Int, text: Bool)]
    let prefix: Int
    let targetCount: Int

    init(imageSlots: [Bool], shapes: [(height: Int, width: Int)], axes: [Int]) throws {
        guard let target = shapes.last, axes.count == 3, axes.allSatisfy({ $0 > 0 && $0 % 2 == 0 }),
              shapes.allSatisfy({ $0.height > 0 && $0.width > 0 }) else { throw LocalImageError.invalid("无效的图像位置配置") }
        targetCount = target.height * target.width
        guard targetCount % 4 == 0 else { throw LocalImageError.invalid("图片宽高必须是 32 的倍数") }
        let slots = imageSlots + Array(repeating: true, count: targetCount / 4)
        var repeated: [Int32] = [], images: [Int32] = [], ids: [Int] = []
        for (index, image) in slots.enumerated() {
            for _ in 0..<(image ? 4 : 1) {
                if image { images.append(Int32(repeated.count)) }
                repeated.append(Int32(index)); ids.append(-1)
            }
        }
        guard shapes.reduce(0, { $0 + $1.height * $1.width }) == images.count else {
            throw LocalImageError.invalid("视觉编码器与 VAE 的参考图尺寸不一致")
        }
        prefix = repeated.count - targetCount
        var coordinates = Array(repeating: Array(repeating: Float(0), count: repeated.count), count: 3)
        var cursor = 0, position = 0, offset = 0
        for (imageID, shape) in shapes.enumerated() {
            let count = shape.height * shape.width
            let start = Int(images[offset])
            guard (0..<count).allSatisfy({ Int(images[offset + $0]) == start + $0 }) else {
                throw LocalImageError.invalid("参考图 token 必须连续")
            }
            for i in cursor..<start {
                for axis in 0..<3 { coordinates[axis][i] = Float(position + i - cursor) }
            }
            position += start - cursor
            for i in 0..<count {
                let index = start + i
                ids[index] = imageID
                coordinates[0][index] = Float(position)
                coordinates[1][index] = Float(i / shape.width - (shape.height - shape.height / 2))
                coordinates[2][index] = Float(i % shape.width - (shape.width - shape.width / 2))
            }
            position += max(shape.height, shape.width); cursor = start + count; offset += count
        }
        var angles: [MLXArray] = []
        for (coordinate, dimension) in zip(coordinates, axes) {
            let frequencies = (0..<(dimension / 2)).map { pow(Float(10000), -Float(2 * $0) / Float(dimension)) }
            angles.append(MLXArray(coordinate)[0..., .newAxis] * MLXArray(frequencies)[.newAxis, 0...])
        }
        let angle = concatenated(angles, axis: -1)
        cos = MLX.cos(angle); sin = MLX.sin(angle)
        repeatIndices = MLXArray(repeated); imageIndices = MLXArray(images)
        let prefixLength = prefix
        targetMask = MLXArray((0..<repeated.count).map { $0 >= prefixLength })
        var pieces: [(Int, Int, Bool)] = [], start = 0
        if prefix > 0 {
            for end in 1...prefix where end == prefix || ids[end] != ids[start] {
                pieces.append((start, end, ids[start] < 0)); start = end
            }
        }
        segments = pieces
    }

    static func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let pairs = x.asType(.float32).reshaped(x.shape.dropLast() + [x.dim(-1) / 2, 2])
        let real = pairs[.ellipsis, 0], imaginary = pairs[.ellipsis, 1]
        let c = cos[.newAxis, .newAxis, 0..., 0...], s = sin[.newAxis, .newAxis, 0..., 0...]
        return stacked([real * c - imaginary * s, imaginary * c + real * s], axis: -1).reshaped(x.shape).asType(x.dtype)
    }
}

final class QwenImage21Transformer {
    let weights: QwenImage21Weights
    let heads: Int, headDim: Int, layers: Int
    let axes: [Int]
    let eps: Float
    private let modulationName: String
    private let timePrefix: String

    init(weights: QwenImage21Weights, config: [String: Any]) throws {
        self.weights = weights
        heads = config["num_attention_heads"] as? Int ?? 32
        headDim = config["attention_head_dim"] as? Int ?? 128
        layers = config["num_layers"] as? Int ?? 32
        axes = config["axes_dims_rope"] as? [Int] ?? [16, 56, 56]
        eps = (config["eps"] as? NSNumber)?.floatValue ?? 1e-6
        guard heads > 0, layers > 0, axes.reduce(0, +) == headDim,
              config["causal_condition"] as? Bool != false else { throw LocalImageError.unsupported("DiT architecture") }
        modulationName = weights.arrays["modulation.0.weight"] == nil ? "modulation.1" : "modulation.0"
        timePrefix = weights.arrays["time_text_embed.linear_1.weight"] == nil
            ? "time_text_embed.timestep_embedder" : "time_text_embed"
    }

    /// Match the pipeline's 0...1000 timestep cast and the transformer's
    /// normalized timestep cast before computing sinusoidal phases in FP32.
    static func timeInput(_ sigma: Float, dtype: DType) -> MLXArray {
        let frequency = exp(-log(MLXArray(Float(10000))) * MLXArray(0..<128).asType(.float32) / 128)
        let timestep = ((MLXArray(sigma) * 1000).asType(dtype) / 1000).asType(dtype)
        let rows = concatenated([timestep.reshaped(1), MLXArray.zeros([1], dtype: dtype)])
        let phase = rows.asType(.float32)[0..., .newAxis] * 1000 * frequency[.newAxis, 0...]
        return concatenated([cos(phase), sin(phase)], axis: -1).asType(dtype)
    }

    private func time(_ timestep: Float, dtype: DType) throws -> MLXArray {
        let input = Self.timeInput(timestep, dtype: dtype)
        let hidden = try weights.linear(input, timePrefix + ".linear_1")
        return try weights.linear(silu(hidden), timePrefix + ".linear_2")
    }

    private func textProjection(_ hidden: MLXArray) throws -> MLXArray {
        let x = hidden.asType(.float32)
        let scale = try weights.get("txt_in.text_norm.weight").asType(.float32) + 1
        let normalized = (x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps) * scale).asType(hidden.dtype)
        return try weights.linear(geluApproximate(weights.linear(normalized, "txt_in.in_layer")), "txt_in.out_layer")
    }

    private func selected(_ rows: MLXArray, mask: MLXArray, targetOnly: Bool = false) -> MLXArray {
        // Cached passes contain only target tokens. Keep the modulation as a
        // singleton row and broadcast in the consumer instead of materializing
        // the same row once per image token.
        if targetOnly { return rows[0..<1, 0...].expandedDimensions(axis: 1) }
        // split() leaves modulation rows noncontiguous. Scalar row indexing
        // uses MLX's general Gather kernel, whose empty index-shape buffers
        // fail Metal API validation in MLX 0.32.3. Range slices preserve the
        // singleton row without Gather and keep the same broadcast semantics.
        return which(mask[.newAxis, 0..., .newAxis],
              rows[0..<1, 0...].expandedDimensions(axis: 1),
              rows[1..<2, 0...].expandedDimensions(axis: 1))
    }

    func predict(latents: MLXArray, text: MLXArray, timestep: Float,
                 layout: QwenImage21Layout, cache: inout [(MLXArray, MLXArray)],
                 control: QwenImage21Control? = nil, controlContext: MLXArray? = nil,
                 controlScale: Float = 1) throws -> MLXArray {
        var controlCache: [(MLXArray, MLXArray)] = []
        if control != nil && controlContext != nil && controlScale > 0 { cache.removeAll() }
        return try predict(latents: latents, text: text, timestep: timestep,
            layout: layout, cache: &cache, control: control,
            controlContext: controlContext, controlScale: controlScale,
            controlCache: &controlCache)
    }

    /// First pass extracts step-independent text/reference K/V; later passes run
    /// only the target image. The caller owns cache lifetime and must match
    /// conditioning and control state before reusing a prefix.
    func predict(latents: MLXArray, text: MLXArray, timestep: Float,
                 layout: QwenImage21Layout, cache: inout [(MLXArray, MLXArray)],
                 control: QwenImage21Control? = nil, controlContext: MLXArray? = nil,
                 controlScale: Float = 1,
                 controlCache: inout [(MLXArray, MLXArray)]) throws -> MLXArray {
        let controlled = control != nil && controlContext != nil && controlScale > 0
        let cached = !cache.isEmpty
        guard !cached || cache.count == layers else { throw LocalImageError.invalid("不完整的图像特征缓存") }
        guard !controlled || (cached ? controlCache.count == control!.blocks : controlCache.isEmpty) else {
            throw LocalImageError.invalid("不完整的 ControlNet 特征缓存")
        }
        var x: MLXArray
        let mask: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray
        if cached {
            x = try weights.linear(latents[0..., (-layout.targetCount)..., 0...], "img_in")
            mask = layout.targetMask[layout.prefix...]
            ropeCos = layout.cos[layout.prefix..., 0...]; ropeSin = layout.sin[layout.prefix..., 0...]
        } else {
            let projected = try textProjection(text)
            let slots = concatenated([projected, MLXArray.zeros([1, layout.targetCount / 4, projected.dim(-1)], dtype: projected.dtype)], axis: 1)
            x = slots[0..., layout.repeatIndices, 0...]
            x[0..., layout.imageIndices, 0...] = try weights.linear(latents, "img_in")
            mask = layout.targetMask; ropeCos = layout.cos; ropeSin = layout.sin
        }
        let embedding = try time(timestep, dtype: x.dtype)
        let modulation = try weights.linear(silu(embedding), modulationName).split(parts: 4, axis: -1)
        let scale1 = selected(modulation[0], mask: mask, targetOnly: cached), gate1 = tanh(selected(modulation[1], mask: mask, targetOnly: cached))
        let scale2 = selected(modulation[2], mask: mask, targetOnly: cached), gate2 = tanh(selected(modulation[3], mask: mask, targetOnly: cached))
        var hints: [MLXArray] = []
        if controlled, let control, let controlContext {
            guard control.blocks == (layers + 1) / 2,
                  controlContext.shape == [1, layout.targetCount, control.inputChannels] else {
                throw LocalImageError.invalid("ControlNet 控制图尺寸或层数不匹配")
            }
            var c: MLXArray
            if cached {
                c = try control.weights.linear(controlContext, "control_img_in").asType(x.dtype)
            } else {
                c = MLXArray.zeros(x.shape, dtype: x.dtype)
                c[0..., layout.prefix..., 0...] = try control.weights.linear(controlContext, "control_img_in").asType(x.dtype)
            }
            c = try control.weights.linear(c, "control_blocks.0.before_proj").asType(x.dtype) + x
            for block in 0..<control.blocks {
                try Task.checkCancellation()
                let output = try runBlock(c, name: "control_blocks.\(block)", blockWeights: control.weights,
                    layout: layout, ropeCos: ropeCos, ropeSin: ropeSin,
                    scale1: scale1, gate1: gate1, scale2: scale2, gate2: gate2,
                    previous: cached ? controlCache[block] : nil)
                c = output.0
                if !cached { controlCache.append(output.1) }
                let hint = try control.weights.linear(c, "control_blocks.\(block).after_proj").asType(x.dtype)
                eval(c, hint)
                hints.append(hint)
            }
        }
        for block in 0..<layers {
            try Task.checkCancellation()
            let output = try runBlock(x, name: "transformer_blocks.\(block)", blockWeights: weights,
                layout: layout, ropeCos: ropeCos, ropeSin: ropeSin,
                scale1: scale1, gate1: gate1, scale2: scale2, gate2: gate2,
                previous: cached ? cache[block] : nil)
            x = output.0
            if !cached { cache.append(output.1) }
            if controlled && block % 2 == 0 { x = x + hints[block / 2] * MLXArray(controlScale).asType(x.dtype) }
            eval(x)
        }
        let scale = try selected(weights.linear(silu(embedding), "norm_out.linear"), mask: mask, targetOnly: cached)
        return try weights.linear(MLXFast.layerNorm(x, eps: eps) * (1 + scale), "proj_out")[0..., (-layout.targetCount)..., 0...]
    }

    /// Shared single-stream block math for the base and Union control chains.
    private func runBlock(_ inputStates: MLXArray, name: String, blockWeights: QwenImage21Weights,
        layout: QwenImage21Layout, ropeCos: MLXArray, ropeSin: MLXArray,
        scale1: MLXArray, gate1: MLXArray, scale2: MLXArray, gate2: MLXArray,
        previous: (MLXArray, MLXArray)?) throws -> (MLXArray, (MLXArray, MLXArray)) {
        var x = inputStates
        let cached = previous != nil
        let input = MLXFast.layerNorm(x, eps: eps) * (1 + scale1)
        let batch = input.dim(0), length = input.dim(1), dim = input.dim(2)
        let q0 = try blockWeights.linear(input, name + ".attn.to_q").reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let k0 = try blockWeights.linear(input, name + ".attn.to_k").reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let v0 = try blockWeights.linear(input, name + ".attn.to_v").reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let q = try QwenImage21Layout.rotate(MLXFast.rmsNorm(q0, weight: blockWeights.get(name + ".attn.norm_q.weight"), eps: eps), cos: ropeCos, sin: ropeSin)
        let k1 = try QwenImage21Layout.rotate(MLXFast.rmsNorm(k0, weight: blockWeights.get(name + ".attn.norm_k.weight"), eps: eps), cos: ropeCos, sin: ropeSin)
        let k: MLXArray, v: MLXArray
        if cached { k = concatenated([previous!.0, k1], axis: 2); v = concatenated([previous!.1, v0], axis: 2) }
        else {
            k = k1; v = v0
        }
        let attention: MLXArray
        let scale = 1 / sqrt(Float(headDim))
        if cached {
            attention = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        } else {
            var pieces: [MLXArray] = []
            for segment in layout.segments + [(layout.prefix, length, false)] {
                let causal: MLXArray? = segment.text ? MLXArray(0..<segment.end)[.newAxis, 0...] .<= MLXArray(segment.start..<segment.end)[0..., .newAxis] : nil
                pieces.append(MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., segment.start..<segment.end, 0...],
                    keys: k[0..., 0..., ..<segment.end, 0...], values: v[0..., 0..., ..<segment.end, 0...], scale: scale, mask: causal))
            }
            attention = concatenated(pieces, axis: 2)
        }
        let result = attention.transposed(0, 2, 1, 3).reshaped(batch, length, dim)
        x = x + gate1 * (try blockWeights.linear(result, name + ".attn.to_out.0"))
        let normalized = MLXFast.layerNorm(x, eps: eps) * (1 + scale2)
        let hidden = try silu(blockWeights.linear(normalized, name + ".img_mlp.gate_layer")) * blockWeights.linear(normalized, name + ".img_mlp.proj")
        x = x + gate2 * (try blockWeights.linear(hidden, name + ".img_mlp.out"))
        if let previous { return (x, previous) }
        return (x, (contiguous(k[0..., 0..., ..<layout.prefix, 0...]),
                    contiguous(v[0..., 0..., ..<layout.prefix, 0...])))
    }

}
