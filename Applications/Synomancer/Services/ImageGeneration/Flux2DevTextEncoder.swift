// Native Mistral hidden-state extraction for FLUX.2-dev.
// Adapted from Hugging Face Transformers/Diffusers (Apache-2.0).
// Copyright 2025 Black Forest Labs and The HuggingFace Team.
// Source pins and licenses: docs/LocalImages.md.
import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

final class Flux2DevTextEncoder {
    static let systemMessage = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation."
    let weights: QwenImage21Weights
    let prefix: String
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let eps: Float
    let theta: Float

    convenience init(directory: URL) throws {
        let c = try LocalImageModelCache.json(directory.appendingPathComponent("config.json"))
        guard c["model_type"] as? String == "mistral3", LocalImageModelCache.supportsFlux2Quantization(c),
              let text = c["text_config"] as? [String: Any],
              text["hidden_size"] as? Int == 5120, text["num_hidden_layers"] as? Int == 40 else {
            throw LocalImageError.unsupported("FLUX.2-dev Mistral encoder (BF16 / MLX affine 8bit)")
        }
        try self.init(weights: QwenImage21Weights(directory: directory), config: text)
    }

    init(weights: QwenImage21Weights, config c: [String: Any]) throws {
        guard let heads = c["num_attention_heads"] as? Int,
              let kvHeads = c["num_key_value_heads"] as? Int,
              let headDim = c["head_dim"] as? Int,
              heads > 0, kvHeads > 0, heads % kvHeads == 0,
              (c["num_hidden_layers"] as? Int ?? 0) >= 30,
              c["sliding_window"] == nil || c["sliding_window"] is NSNull,
              c["use_qk_norm"] as? Bool != true else {
            throw LocalImageError.unsupported("Mistral full-attention text configuration")
        }
        let rope = (c["rope_parameters"] ?? c["rope_scaling"]) as? [String: Any] ?? [:]
        guard (rope["rope_type"] as? String ?? "default") == "default" else {
            throw LocalImageError.unsupported("Mistral scaled RoPE")
        }
        guard let prefix = ["model.language_model.", "language_model.model.", "language_model.", "model."].first(where: {
            weights.arrays[$0 + "embed_tokens.weight"] != nil
        }) else { throw LocalImageError.missing("Mistral text embedding") }
        self.weights = weights; self.prefix = prefix
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        eps = (c["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-5
        theta = ((c["rope_theta"] ?? rope["rope_theta"]) as? NSNumber)?.floatValue ?? 1_000_000_000
        // No vision pass or vocabulary projection is used by Flux2Pipeline.
        // Keep only the 30 layers needed by intermediate states 10/20/30.
        weights.arrays = weights.arrays.filter { name, _ in
            if ["weight", "scales", "biases"].contains(where: { name == prefix + "embed_tokens." + $0 }) { return true }
            guard name.hasPrefix(prefix + "layers.") else { return false }
            let index = name.dropFirst((prefix + "layers.").count).split(separator: ".").first
            return index.flatMap { Int($0) }.map { $0 < 30 } ?? false
        }.mapValues { $0.dtype == .float16 ? $0.asType(.bfloat16) : $0 }
    }

    static func tokens(_ prompt: String, directory: URL) async throws -> ([Int], Int) {
        // This LlamaTokenizerFast export uses byte-level BPE, not SentencePiece.
        // Read the exported tokenizer.json unchanged, without routing this older
        // tokenizer class name through SentencePiece/Metaspace overrides.
        guard let config = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))) as? [NSString: Any],
              let data = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))) as? [NSString: Any] else {
            throw LocalImageError.missing("Mistral tokenizer JSON dictionary")
        }
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: .init(config), tokenizerData: .init(data))
        // Exact text-only rendering of the official processor chat template;
        // no assistant generation prompt, image tokens, or automatic upsampling.
        let raw = "<s>[SYSTEM_PROMPT]" + systemMessage + "[/SYSTEM_PROMPT][INST]"
            + prompt.replacingOccurrences(of: "[IMG]", with: "") + "[/INST]"
        var tokens = Array(tokenizer.encode(text: raw, addSpecialTokens: false).prefix(512))
        let count = tokens.count
        guard let pad = tokenizer.convertTokenToId("<pad>") else { throw LocalImageError.missing("Mistral padding token") }
        tokens = Array(repeating: pad, count: 512 - count) + tokens
        return (tokens, count)
    }

    func encode(_ tokens: [Int], validCount: Int) throws -> MLXArray {
        guard !tokens.isEmpty, validCount > 0, validCount <= tokens.count else {
            throw LocalImageError.invalid("Mistral prompt token mask is empty or invalid")
        }
        let count = tokens.count
        let padding = count - validCount
        var x = try weights.embedding(MLXArray(tokens), prefix + "embed_tokens").expandedDimensions(axis: 0)
        let mask = MLXArray((0..<count).flatMap { row in
            (0..<count).map { col -> Float in
                // Empty leading-pad queries have zero SDPA output in PyTorch.
                // Give them a safe diagonal, then explicitly zero their output.
                let visible = row < padding ? col == row : (col <= row && col >= padding)
                return visible ? 0 : -.infinity
            }
        }).reshaped(1, 1, count, count).asType(x.dtype)
        let validQueries = MLXArray((0..<count).map { Float($0 >= padding ? 1 : 0) }).reshaped(1, count, 1).asType(x.dtype)
        var states: [MLXArray] = []
        func norm(_ input: MLXArray, _ name: String) throws -> MLXArray {
            try MLXFast.rmsNorm(input, weight: weights.get(name + ".weight"), eps: eps)
        }
        for layer in 0..<30 {
            try Task.checkCancellation()
            let p = prefix + "layers.\(layer)", a = p + ".self_attn"
            let n = try norm(x, p + ".input_layernorm")
            var q = try weights.linear(n, a + ".q_proj").reshaped(1, count, heads, headDim).transposed(0, 2, 1, 3)
            var k = try weights.linear(n, a + ".k_proj").reshaped(1, count, kvHeads, headDim).transposed(0, 2, 1, 3)
            let v = try weights.linear(n, a + ".v_proj").reshaped(1, count, kvHeads, headDim).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: 0)
            let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 1 / sqrt(Float(headDim)), mask: .array(mask))
                .transposed(0, 2, 1, 3).reshaped(1, count, heads * headDim) * validQueries
            x = try x + weights.linear(attended, a + ".o_proj")
            let m = try norm(x, p + ".post_attention_layernorm")
            let gated = try silu(weights.linear(m, p + ".mlp.gate_proj")) * weights.linear(m, p + ".mlp.up_proj")
            x = try x + weights.linear(gated, p + ".mlp.down_proj")
            eval(x)
            if [10, 20, 30].contains(layer + 1) { states.append(x) }
        }
        return concatenated(states, axis: -1)
    }
}
