// Native Swift/MLX adaptation of Transformers CLIP and T5 encoders (Apache-2.0).
// Copyright The HuggingFace Team. Sources and license: docs/LocalImages.md.
import Foundation
import MLX
import MLXNN
import Tokenizers

final class Flux1CLIPTokenizer {
    private struct Pair: Hashable { let a: String; let b: String }
    private let vocabulary: [String: Int]
    private let ranks: [Pair: Int]
    private let bytes: [UInt8: String]
    let bos: Int, eos: Int

    init(directory: URL) throws {
        vocabulary = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        guard let bos = vocabulary["<|startoftext|>"], let eos = vocabulary["<|endoftext|>"] else {
            throw LocalImageError.missing("CLIP special tokens")
        }
        self.bos = bos; self.eos = eos
        let merges = try String(contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8)
            .split(separator: "\n").filter { !$0.hasPrefix("#") }.map { $0.split(separator: " ") }
        var ranks: [Pair: Int] = [:]
        for (index, pair) in merges.enumerated() where pair.count == 2 {
            ranks[Pair(a: String(pair[0]), b: String(pair[1]))] = index
        }
        self.ranks = ranks
        var visible = Array(33...126) + Array(161...172) + Array(174...255)
        var codepoints = visible, extra = 0
        for byte in 0...255 where !visible.contains(byte) {
            visible.append(byte); codepoints.append(256 + extra); extra += 1
        }
        bytes = Dictionary(uniqueKeysWithValues: zip(visible, codepoints).map { (UInt8($0), String(UnicodeScalar($1)!)) })
    }

    func encode(_ text: String, length: Int = 77) throws -> ([Int], Int) {
        // CLIP operates on UTF-8 bytes; character-only BPE drops non-ASCII prompts.
        let clean = text.precomposedStringWithCanonicalMapping.lowercased()
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        let regex = try NSRegularExpression(pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#)
        var tokens = [bos]
        for match in regex.matches(in: clean, range: NSRange(clean.startIndex..., in: clean)) {
            let piece = String(clean[Range(match.range, in: clean)!])
            if piece == "<|startoftext|>" { tokens.append(bos); continue }
            if piece == "<|endoftext|>" { tokens.append(eos); continue }
            var parts = piece.utf8.map { bytes[$0]! }
            parts[parts.count - 1] += "</w>"
            while parts.count > 1 {
                let candidates = zip(parts, parts.dropFirst()).map { Pair(a: $0, b: $1) }
                guard let pair = candidates.min(by: { (ranks[$0] ?? Int.max) < (ranks[$1] ?? Int.max) }), ranks[pair] != nil else { break }
                var merged: [String] = [], i = 0
                while i < parts.count {
                    if i + 1 < parts.count && parts[i] == pair.a && parts[i + 1] == pair.b {
                        merged.append(pair.a + pair.b); i += 2
                    } else { merged.append(parts[i]); i += 1 }
                }
                parts = merged
            }
            for part in parts {
                guard let id = vocabulary[part] else { throw LocalImageError.missing("CLIP BPE token " + part) }
                tokens.append(id)
            }
        }
        tokens = Array(tokens.prefix(length - 1)) + [eos]
        // CLIP's configured eos=2 uses the historical highest-token-id pooling.
        let pooled = tokens.firstIndex(of: tokens.max()!)!
        tokens += Array(repeating: eos, count: length - tokens.count)
        return (tokens, pooled)
    }
}

final class Flux1TextEncoder {
    let weights: QwenImage21Weights
    let config: [String: Any]
    init(directory: URL) throws {
        weights = try QwenImage21Weights(directory: directory)
        config = try LocalImageModelCache.json(directory.appendingPathComponent("config.json"))
    }

    static func t5Tokens(_ prompt: String, tokenizer: any Tokenizers.Tokenizer, length: Int = 512) throws -> [Int] {
        guard let eos = tokenizer.convertTokenToId("</s>"), let pad = tokenizer.convertTokenToId("<pad>") else {
            throw LocalImageError.missing("T5 special tokens")
        }
        // swift-transformers emits a standalone metaspace for empty input.
        // Transformers emits only EOS; don't create an extra content token.
        let content = prompt.isEmpty ? [] : tokenizer.encode(text: prompt, addSpecialTokens: false)
        var tokens = Array(content.prefix(length - 1)) + [eos]
        tokens += Array(repeating: pad, count: length - tokens.count)
        return tokens
    }

    func clip(_ tokens: [Int], pooledIndex: Int) throws -> MLXArray {
        let dim = config["hidden_size"] as? Int ?? 768
        let heads = config["num_attention_heads"] as? Int ?? 12
        let layers = config["num_hidden_layers"] as? Int ?? 12
        let eps = (config["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-5
        let count = tokens.count
        let root = weights.arrays["text_model.embeddings.token_embedding.weight"] != nil ? "text_model." : ""
        var x = try weights.get(root + "embeddings.token_embedding.weight")[MLXArray(tokens)].expandedDimensions(axis: 0)
            + weights.get(root + "embeddings.position_embedding.weight")[..<count].expandedDimensions(axis: 0)
        let mask = MLXArray((0..<count).flatMap { row in (0..<count).map { $0 <= row ? Float(0) : -Float.infinity } })
            .reshaped(1, 1, count, count).asType(x.dtype)
        func norm(_ x: MLXArray, _ name: String) throws -> MLXArray {
            try MLXFast.layerNorm(x, weight: weights.get(name + ".weight"), bias: weights.get(name + ".bias"), eps: eps)
        }
        func splitHeads(_ x: MLXArray) -> MLXArray { x.reshaped(1, count, heads, dim / heads).transposed(0, 2, 1, 3) }
        for layer in 0..<layers {
            try Task.checkCancellation()
            let p = root + "encoder.layers.\(layer)", a = p + ".self_attn"
            let n = try norm(x, p + ".layer_norm1")
            let q = try splitHeads(weights.linear(n, a + ".q_proj"))
            let k = try splitHeads(weights.linear(n, a + ".k_proj"))
            let v = try splitHeads(weights.linear(n, a + ".v_proj"))
            let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 1 / sqrt(Float(dim / heads)), mask: .array(mask)).transposed(0, 2, 1, 3).reshaped(1, count, dim)
            x = try x + weights.linear(attended, a + ".out_proj")
            let hidden = try weights.linear(norm(x, p + ".layer_norm2"), p + ".mlp.fc1")
            x = try x + weights.linear(hidden * sigmoid(1.702 * hidden), p + ".mlp.fc2")
            eval(x)
        }
        return try norm(x, root + "final_layer_norm")[0..., pooledIndex, 0...]
    }

    static func relativeBucket(_ relative: Int, buckets: Int, distance: Int) -> Int {
        let half = buckets / 2, exact = half / 2, n = abs(relative)
        let value = n < exact ? n : min(half - 1,
            exact + Int(log(Double(n) / Double(exact)) / log(Double(distance) / Double(exact)) * Double(half - exact)))
        return (relative > 0 ? half : 0) + value
    }

    func t5(_ tokens: [Int]) throws -> MLXArray {
        let heads = config["num_heads"] as? Int ?? 64, headDim = config["d_kv"] as? Int ?? 64
        let layers = config["num_layers"] as? Int ?? 24
        let eps = (config["layer_norm_epsilon"] as? NSNumber)?.floatValue ?? 1e-6
        let buckets = config["relative_attention_num_buckets"] as? Int ?? 32
        let distance = config["relative_attention_max_distance"] as? Int ?? 128
        let count = tokens.count
        var x = try weights.get(weights.arrays["shared.weight"] == nil ? "encoder.embed_tokens.weight" : "shared.weight")[MLXArray(tokens)].expandedDimensions(axis: 0)
        let positions = MLXArray((0..<count).flatMap { row in (0..<count).map {
            Int32(Self.relativeBucket($0 - row, buckets: buckets, distance: distance))
        } }).reshaped(count, count)
        let bias = try weights.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")[positions]
            .transposed(2, 0, 1).expandedDimensions(axis: 0).asType(x.dtype)
        func norm(_ input: MLXArray, _ name: String) throws -> MLXArray {
            let y = input.asType(.float32)
            return try (y * rsqrt((y * y).mean(axis: -1, keepDims: true) + eps)).asType(input.dtype) * weights.get(name + ".weight")
        }
        func splitHeads(_ x: MLXArray) -> MLXArray { x.reshaped(1, count, heads, headDim).transposed(0, 2, 1, 3) }
        for layer in 0..<layers {
            try Task.checkCancellation()
            let p = "encoder.block.\(layer)", a = p + ".layer.0.SelfAttention"
            let n = try norm(x, p + ".layer.0.layer_norm")
            let q = try splitHeads(weights.linear(n, a + ".q")), k = try splitHeads(weights.linear(n, a + ".k"))
            let v = try splitHeads(weights.linear(n, a + ".v"))
            // T5 deliberately does not apply the usual 1/sqrt(headDim) scale;
            // padding tokens also remain visible, as in FluxPipeline.encode_prompt.
            let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 1, mask: .array(bias)).transposed(0, 2, 1, 3).reshaped(1, count, heads * headDim)
            x = try x + weights.linear(attended, a + ".o")
            let m = try norm(x, p + ".layer.1.layer_norm"), ff = p + ".layer.1.DenseReluDense"
            let gated = try geluApproximate(weights.linear(m, ff + ".wi_0")) * weights.linear(m, ff + ".wi_1")
            x = try x + weights.linear(gated, ff + ".wo")
            eval(x)
        }
        return try norm(x, "encoder.final_layer_norm")
    }
}
