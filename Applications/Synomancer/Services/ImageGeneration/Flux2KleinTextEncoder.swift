// Qwen3 intermediate features for FLUX.2 Klein; see docs/LocalImages.md.
// Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
// Copyright 2026 The HuggingFace Team. Apache-2.0.
// MFLUX portions: Copyright (c) 2026 Filip Strand. MIT.
import Foundation
import MLX
import MLXNN
import Tokenizers

final class Flux2KleinTextEncoder {
    let weights: QwenImage21Weights
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let eps: Float
    let theta: Float

    convenience init(directory: URL) throws {
        let c = try LocalImageModelCache.json(directory.appendingPathComponent("config.json"))
        guard c["model_type"] as? String == "qwen3",
              c["hidden_size"] as? Int == 4096,
              c["num_hidden_layers"] as? Int == 36 else {
            throw LocalImageError.unsupported("Klein 9B Qwen3 encoder")
        }
        try self.init(weights: QwenImage21Weights(directory: directory), config: c)
    }

    init(weights: QwenImage21Weights, config c: [String: Any]) throws {
        // The community checkpoint stores FP16 weights, while the reference
        // pipeline runs in BF16. Keep its exponent range for intermediate states.
        weights.arrays = weights.arrays.mapValues {
            $0.dtype == .float16 ? $0.asType(.bfloat16) : $0
        }
        heads = c["num_attention_heads"] as? Int ?? 32
        kvHeads = c["num_key_value_heads"] as? Int ?? 8
        headDim = c["head_dim"] as? Int ?? 128
        eps = (c["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        let rope = c["rope_parameters"] as? [String: Any] ?? [:]
        theta = ((c["rope_theta"] ?? rope["rope_theta"]) as? NSNumber)?.floatValue ?? 1_000_000
        self.weights = weights
        // Only intermediate states 9, 18, 27 are consumed. Release the unused
        // vocabulary output projection and upper decoder layers before evaluation.
        weights.arrays = weights.arrays.filter { key, _ in
            if key == "model.embed_tokens.weight" { return true }
            let parts = key.split(separator: ".")
            return parts.count > 3 && parts[0] == "model" && parts[1] == "layers"
                && (Int(parts[2]).map { $0 < 27 } ?? false)
        }
    }

    static func tokens(_ prompt: String, directory: URL) async throws -> ([Int], Int) {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let formatted = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        var tokens = Array(tokenizer.encode(text: formatted, addSpecialTokens: false).prefix(512))
        let count = tokens.count
        guard let padding = tokenizer.convertTokenToId("<|endoftext|>") else {
            throw LocalImageError.missing("Qwen3 padding token")
        }
        tokens += Array(repeating: padding, count: 512 - count)
        return (tokens, count)
    }

    func encode(_ tokens: [Int], validCount: Int) throws -> MLXArray {
        let count = tokens.count
        var x = try weights.get("model.embed_tokens.weight")[MLXArray(tokens)].expandedDimensions(axis: 0)
        let mask = MLXArray((0..<count).flatMap { row in
            (0..<count).map { col -> Float in col <= row && col < validCount ? 0 : -.infinity }
        }).reshaped(1, 1, count, count).asType(x.dtype)
        var features: [MLXArray] = []
        func norm(_ input: MLXArray, _ name: String) throws -> MLXArray {
            try MLXFast.rmsNorm(input, weight: weights.get(name + ".weight"), eps: eps)
        }
        for layer in 0..<27 {
            try Task.checkCancellation()
            let p = "model.layers.\(layer)"
            let a = p + ".self_attn"
            let n = try norm(x, p + ".input_layernorm")
            var q = try norm(weights.linear(n, a + ".q_proj").reshaped(1, count, heads, headDim), a + ".q_norm").transposed(0, 2, 1, 3)
            var k = try norm(weights.linear(n, a + ".k_proj").reshaped(1, count, kvHeads, headDim), a + ".k_norm").transposed(0, 2, 1, 3)
            let v = try weights.linear(n, a + ".v_proj").reshaped(1, count, kvHeads, headDim).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: 0)
            let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 1 / sqrt(Float(headDim)), mask: .array(mask)).transposed(0, 2, 1, 3).reshaped(1, count, heads * headDim)
            x = try x + weights.linear(attended, a + ".o_proj")
            let m = try norm(x, p + ".post_attention_layernorm")
            let gated = try silu(weights.linear(m, p + ".mlp.gate_proj")) * weights.linear(m, p + ".mlp.up_proj")
            x = try x + weights.linear(gated, p + ".mlp.down_proj")
            eval(x)
            if [9, 18, 27].contains(layer + 1) { features.append(x) }
        }
        return concatenated(features, axis: -1)
    }
}
