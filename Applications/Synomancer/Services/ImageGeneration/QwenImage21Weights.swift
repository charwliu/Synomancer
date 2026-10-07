// Native Swift adaptation of Qwen/Diffusers and MFLUX; see docs/LocalImages.md.
// Copyright 2026 The Qwen Team and The HuggingFace Team. Apache-2.0.
import Foundation
import MLX

enum LocalImageError: LocalizedError {
    case unavailable, missing(String), unsupported(String), invalid(String), nonfinite
    var errorDescription: String? {
        switch self {
        case .unavailable: "HF_HOME 中没有完整的 Qwen-Image-2.1 模型。请添加官方模型或包含视觉编码器的 MLX 量化模型。"
        case .missing(let name): "图像模型缺少或损坏的文件/参数：\(name)"
        case .unsupported(let name): "暂不支持此图像模型配置：\(name)"
        case .invalid(let reason): reason
        case .nonfinite: "图像推理产生了无效数值，未保存结果。"
        }
    }
}

struct LocalImageModel: Identifiable, Sendable {
    let id: String
    let directory: URL
    let bytes: UInt64
    let quantized: Bool
}

enum LocalImageModelCache {
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
                guard try json(directory.appendingPathComponent("model_index.json"))["_class_name"] as? String == "QwenImage21Pipeline" else { return }
                let config = try json(directory.appendingPathComponent("transformer/config.json"))
                let parts = ["transformer", "text_encoder", "vae"]
                let files = try parts.flatMap { try shards(directory.appendingPathComponent($0)) }
                for name in ["processor/tokenizer.json", "processor/tokenizer_config.json", "processor/preprocessor_config.json", "scheduler/scheduler_config.json"] {
                    guard FileManager.default.isReadableFile(atPath: directory.appendingPathComponent(name).path) else { return }
                }
                // Text-only exports cannot edit photographs; retain the vision tower.
                let names = try tensorNames(directory.appendingPathComponent("text_encoder"))
                guard names.contains(where: { $0.hasPrefix("vision_tower.") || $0.hasPrefix("model.visual.") }) else { return }
                let bytes = files.reduce(UInt64(0)) { $0 + UInt64((try? $1.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
                if result[id] == nil {
                    result[id] = .init(id: id, directory: directory, bytes: bytes, quantized: config["quantization"] != nil)
                }
            } catch { /* incomplete snapshots are not runnable */ }
        }
        add(root, id: root.lastPathComponent)
        for hub in [root, root.appendingPathComponent("hub")] {
            let repositories = hub.lastPathComponent.hasPrefix("models--") ? [hub] :
                ((try? FileManager.default.contentsOfDirectory(at: hub, includingPropertiesForKeys: nil)) ?? [])
            for repo in repositories where repo.lastPathComponent.hasPrefix("models--") && repo.lastPathComponent.lowercased().contains("qwen-image-2.1") {
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
