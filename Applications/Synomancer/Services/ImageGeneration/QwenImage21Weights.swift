// Native Swift adaptation of Qwen/Diffusers and MFLUX; see docs/LocalImages.md.
// Copyright 2026 The Qwen Team and The HuggingFace Team. Apache-2.0.
import Foundation
import MLX

enum LocalImageError: LocalizedError {
    case unavailable, missing(String), unsupported(String), invalid(String), nonfinite
    var errorDescription: String? {
        switch self {
        case .unavailable: "HF_HOME 中没有完整的 Qwen-Image-2.1、FLUX.1-dev、FLUX.2-dev 或 FLUX.2 Klein 9B 模型。"
        case .missing(let name): "图像模型缺少或损坏的文件/参数：\(name)"
        case .unsupported(let name): "暂不支持此图像模型配置：\(name)"
        case .invalid(let reason): reason
        case .nonfinite: "图像推理产生了无效数值，未保存结果。"
        }
    }
}

enum LocalImageFamily: String, Sendable {
    case qwenImage21 = "Qwen-Image-2.1"
    case flux1Dev = "FLUX.1-dev"
    case flux2Klein9B = "FLUX.2 Klein 9B"
    case flux2Dev = "FLUX.2-dev"
    var defaultSteps: Int { self == .flux2Klein9B ? 4 : (self == .flux1Dev || self == .flux2Dev) ? 28 : 40 }
    var defaultGuidance: Double { self == .flux2Dev ? 4 : self == .flux2Klein9B ? 1 : 3.5 }
    var defaultTrueCFGScale: Double { self == .flux1Dev ? 1.1 : 2 }
    var supportsNegativePrompt: Bool { self == .qwenImage21 || self == .flux1Dev }
    var supportsControl: Bool { self == .qwenImage21 }
    var supportsReference: Bool { self != .flux1Dev }
}

struct LocalImageModel: Identifiable, Sendable {
    let id: String
    let directory: URL
    let bytes: UInt64
    let quantized: Bool
    var family: LocalImageFamily = .qwenImage21
}

enum LocalImageModelCache {
    /// This path supports native MLX affine Q8, not GGUF or other quantizers.
    static func supportsFlux2Quantization(_ config: [String: Any]) -> Bool {
        guard config["quantization_config"] == nil else { return false }
        guard let value = config["quantization"] else { return true }
        guard let q = value as? [String: Any],
              q["group_size"] == nil || q["group_size"] is Int,
              q["mode"] == nil || q["mode"] is String else { return false }
        return q["bits"] as? Int == 8
            && [32, 64, 128].contains(q["group_size"] as? Int ?? 64)
            && (q["mode"] as? String ?? "affine") == "affine"
    }

    static func json(_ file: URL) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any] else {
            throw LocalImageError.missing(file.lastPathComponent)
        }
        return value
    }

    static func shards(_ directory: URL) throws -> [URL] {
        let indexes = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasSuffix(".safetensors.index.json") }
        let files: [URL]
        if let index = indexes.first {
            guard let map = try json(index)["weight_map"] as? [String: String], !map.isEmpty,
                  map.values.allSatisfy({ !$0.contains("/") && $0.hasSuffix(".safetensors") }) else {
                throw LocalImageError.missing(index.path)
            }
            files = Set(map.values).sorted().map { directory.appendingPathComponent($0) }
        } else {
            files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        }
        guard !files.isEmpty, files.allSatisfy({ FileManager.default.isReadableFile(atPath: $0.path)
            && ((try? $0.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])).map {
                $0.isRegularFile == true && ($0.fileSize ?? 0) > 0
            } ?? false) }) else {
            throw LocalImageError.missing(directory.lastPathComponent + " 权重分片")
        }
        return files
    }

    private static func tensorNames(_ directory: URL) throws -> Set<String> {
        let indexes = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasSuffix(".safetensors.index.json") }
        if let index = indexes.first, let map = try json(index)["weight_map"] as? [String: String] {
            return Set(map.keys)
        }
        // Small safetensors headers only; never load model tensors while listing.
        var names: Set<String> = []
        for shard in try shards(directory) {
            let handle = try FileHandle(forReadingFrom: shard)
            defer { try? handle.close() }
            guard let size = try handle.read(upToCount: 8), size.count == 8 else { throw LocalImageError.missing(shard.path) }
            let length = size.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
            guard length > 0, length <= 16 * 1024 * 1024,
                  let header = try handle.read(upToCount: Int(length)), header.count == Int(length),
                  let entries = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
                throw LocalImageError.missing(shard.path)
            }
            names.formUnion(entries.keys.filter { $0 != "__metadata__" })
        }
        return names
    }

    static func scan(_ root: URL) throws -> [LocalImageModel] {
        var result: [String: LocalImageModel] = [:]
        func add(_ directory: URL, id: String) {
            do {
                let index = try json(directory.appendingPathComponent("model_index.json"))
                let family: LocalImageFamily
                switch index["_class_name"] as? String {
                case "QwenImage21Pipeline": family = .qwenImage21
                case "FluxPipeline": family = .flux1Dev
                case "Flux2Pipeline": family = .flux2Dev
                case "Flux2KleinPipeline" where index["is_distilled"] as? Bool == true: family = .flux2Klein9B
                default: return
                }
                let config = try json(directory.appendingPathComponent("transformer/config.json"))
                let parts = ["transformer", "text_encoder", "vae"] + (family == .flux1Dev ? ["text_encoder_2"] : [])
                let files = try parts.flatMap { try shards(directory.appendingPathComponent($0)) }
                if family == .flux1Dev {
                    let clip = try json(directory.appendingPathComponent("text_encoder/config.json"))
                    let t5 = try json(directory.appendingPathComponent("text_encoder_2/config.json"))
                    let vae = try json(directory.appendingPathComponent("vae/config.json"))
                    guard config["quantization"] == nil, clip["quantization"] == nil, t5["quantization"] == nil,
                          config["guidance_embeds"] as? Bool == true,
                          config["in_channels"] as? Int == 64,
                          config["joint_attention_dim"] as? Int == 4096,
                          config["pooled_projection_dim"] as? Int == 768,
                          config["num_layers"] as? Int == 19,
                          config["num_single_layers"] as? Int == 38,
                          config["num_attention_heads"] as? Int == 24,
                          config["attention_head_dim"] as? Int == 128,
                          clip["model_type"] as? String == "clip_text_model",
                          clip["hidden_size"] as? Int == 768,
                          t5["model_type"] as? String == "t5",
                          t5["d_model"] as? Int == 4096,
                          vae["latent_channels"] as? Int == 16 else { return }
                }
                if family == .flux2Klein9B {
                    guard config["joint_attention_dim"] as? Int == 12288,
                          config["num_layers"] as? Int == 8,
                          config["num_single_layers"] as? Int == 24,
                          config["guidance_embeds"] as? Bool == false else { return }
                }
                if family == .flux2Dev {
                    let encoder = try json(directory.appendingPathComponent("text_encoder/config.json"))
                    let text = encoder["text_config"] as? [String: Any] ?? [:]
                    let vae = try json(directory.appendingPathComponent("vae/config.json"))
                    guard supportsFlux2Quantization(config), supportsFlux2Quantization(encoder),
                          config["in_channels"] as? Int == 128,
                          config["joint_attention_dim"] as? Int == 15360,
                          config["num_attention_heads"] as? Int == 48,
                          config["attention_head_dim"] as? Int == 128,
                          config["num_layers"] as? Int == 8,
                          config["num_single_layers"] as? Int == 48,
                          (config["guidance_embeds"] as? Bool ?? true),
                          encoder["model_type"] as? String == "mistral3",
                          text["hidden_size"] as? Int == 5120,
                          text["num_hidden_layers"] as? Int == 40,
                          vae["latent_channels"] as? Int == 32 else { return }
                }
                let required: [String]
                switch family {
                case .qwenImage21:
                    required = ["processor/tokenizer.json", "processor/tokenizer_config.json", "processor/preprocessor_config.json"]
                case .flux1Dev:
                    required = ["tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
                                "tokenizer_2/tokenizer.json", "tokenizer_2/tokenizer_config.json", "text_encoder_2/config.json"]
                case .flux2Klein9B, .flux2Dev:
                    required = ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
                }
                for name in required + ["scheduler/scheduler_config.json", "text_encoder/config.json", "vae/config.json"] {
                    guard FileManager.default.isReadableFile(atPath: directory.appendingPathComponent(name).path) else { return }
                }
                // Text-only exports cannot edit photographs; retain the vision tower.
                let names = try tensorNames(directory.appendingPathComponent("text_encoder"))
                if family == .qwenImage21 {
                    guard names.contains(where: { $0.hasPrefix("vision_tower.") || $0.hasPrefix("model.visual.") }) else { return }
                }
                let bytes = files.reduce(UInt64(0)) { $0 + UInt64((try? $1.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
                if result[id] == nil {
                    result[id] = .init(id: id, directory: directory, bytes: bytes, quantized: config["quantization"] != nil, family: family)
                }
            } catch { /* incomplete snapshots are not runnable */ }
        }
        add(root, id: root.lastPathComponent)
        for hub in [root, root.appendingPathComponent("hub")] {
            let repositories = hub.lastPathComponent.hasPrefix("models--") ? [hub] :
                ((try? FileManager.default.contentsOfDirectory(at: hub, includingPropertiesForKeys: nil)) ?? [])
            for repo in repositories where repo.lastPathComponent.hasPrefix("models--") && ["qwen-image-2.1", "flux.1-dev", "flux.2-dev", "flux.2-klein-9b"].contains(where: { repo.lastPathComponent.lowercased().contains($0) }) {
                let id = String(repo.lastPathComponent.dropFirst(8)).replacingOccurrences(of: "--", with: "/")
                var snapshots: [URL] = []
                if let revision = try? String(contentsOf: repo.appendingPathComponent("refs/main"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) {
                    snapshots.append(repo.appendingPathComponent("snapshots/" + revision))
                }
                snapshots += (try? FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("snapshots"), includingPropertiesForKeys: nil)) ?? []
                for snapshot in snapshots { add(snapshot, id: id) }
            }
        }
        return result.values.sorted { $0.bytes == $1.bytes ? $0.id < $1.id : $0.bytes < $1.bytes }
    }
}

/// Component arrays are actor-owned. Packed affine weights stay packed on Metal.
final class QwenImage21Weights {
    var arrays: [String: MLXArray]
    let bits: Int
    let groupSize: Int
    let native: Bool

    init(directory: URL) throws {
        let config = try LocalImageModelCache.json(directory.appendingPathComponent("config.json"))
        let quantization = config["quantization"] as? [String: Any] ?? [:]
        bits = quantization["bits"] as? Int ?? 4
        groupSize = quantization["group_size"] as? Int ?? 64
        if !quantization.isEmpty && (quantization["mode"] as? String ?? "affine") != "affine" {
            throw LocalImageError.unsupported("quantization mode")
        }
        arrays = [:]
        var isNative = config["mlx_format"] as? Bool ?? false
        for shard in try LocalImageModelCache.shards(directory) {
            try Task.checkCancellation()
            let (values, metadata) = try MLX.loadArraysAndMetadata(url: shard)
            isNative = isNative || metadata["format"] == "mlx"
            for (name, value) in values {
                guard arrays[name] == nil else { throw LocalImageError.missing("重复参数 " + name) }
                arrays[name] = value
            }
        }
        native = isNative
    }

    init(arrays: [String: MLXArray], bits: Int = 4, groupSize: Int = 64) {
        self.arrays = arrays; self.bits = bits; self.groupSize = groupSize; native = true
    }

    func get(_ name: String) throws -> MLXArray {
        guard let value = arrays[name] else { throw LocalImageError.missing(name) }
        return value
    }

    func embedding(_ tokens: MLXArray, _ name: String) throws -> MLXArray {
        let weight = try get(name + ".weight")
        guard weight.ndim == 2 else { throw LocalImageError.unsupported(name + " embedding shape") }
        if let scales = arrays[name + ".scales"] {
            guard [2, 4, 8].contains(bits), [32, 64, 128].contains(groupSize),
                  weight.dtype == .uint32, scales.ndim == 2,
                  scales.dim(0) == weight.dim(0),
                  scales.dim(1) * groupSize == weight.dim(1) * (32 / bits),
                  let biases = arrays[name + ".biases"], biases.shape == scales.shape else {
                throw LocalImageError.unsupported(name + " packed embedding")
            }
            // Gather first: never expand the entire vocabulary table in memory.
            return MLX.dequantized(weight[tokens], scales: scales[tokens], biases: biases[tokens],
                groupSize: groupSize, bits: bits)
        }
        guard weight.dtype != .uint32 else { throw LocalImageError.missing(name + ".scales") }
        return weight[tokens]
    }

    func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let weight = try get(name + ".weight")
        guard weight.ndim == 2 else { throw LocalImageError.unsupported(name + " matrix shape") }
        var result: MLXArray
        if let scales = arrays[name + ".scales"] {
            guard [2, 4, 8].contains(bits), groupSize > 0, weight.dtype == .uint32,
                  x.dim(-1) == weight.dim(-1) * (32 / bits), scales.ndim == 2,
                  scales.dim(0) == weight.dim(0), scales.dim(1) * groupSize == x.dim(-1),
                  arrays[name + ".biases"]?.shape == scales.shape else {
                throw LocalImageError.unsupported(name + " packed weight")
            }
            result = MLX.quantizedMM(x, weight, scales: scales, biases: arrays[name + ".biases"],
                groupSize: groupSize, bits: bits)
        } else {
            guard x.dim(-1) == weight.dim(-1) else { throw LocalImageError.unsupported(name + " input dimensions") }
            result = matmul(x, weight.T)
        }
        if let bias = arrays[name + ".bias"] { result = result + bias }
        return result
    }
}
