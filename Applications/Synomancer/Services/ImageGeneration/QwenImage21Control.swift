// Copyright (c) Alibaba, Inc. and its affiliates. Apache-2.0.
// Swift/MLX adaptation of VideoX-Fun's Qwen-Image-2.1 Union control chain.
// See docs/LocalImages.md for source revision and model license.
import Foundation
import MLX

struct LocalImageControlModel: Sendable {
    let file: URL
    let bytes: UInt64
}

enum LocalImageControlCache {
    static let repository = "models--alibaba-pai--Qwen-Image-2.1-Fun-Controlnet-Union"
    static let filename = "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"

    static func find(_ root: URL) -> LocalImageControlModel? {
        for hub in [root, root.appendingPathComponent("hub")] {
            let repo = hub.lastPathComponent == repository ? hub : hub.appendingPathComponent(repository)
            var folders = [root]
            if let revision = try? String(contentsOf: repo.appendingPathComponent("refs/main"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
               !revision.contains("/"), !revision.isEmpty {
                folders.append(repo.appendingPathComponent("snapshots/" + revision))
            }
            folders += (try? FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("snapshots"), includingPropertiesForKeys: nil)) ?? []
            for folder in folders {
                let file = folder.appendingPathComponent(filename)
                guard let values = try? file.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                      values.isRegularFile == true, let bytes = values.fileSize, bytes > 0,
                      let header = try? header(file),
                      (header["control_img_in.weight"] as? [String: Any])?["shape"] as? [Int] == [4096, 129],
                      header["control_blocks.15.after_proj.weight"] != nil else { continue }
                return .init(file: file, bytes: UInt64(bytes))
            }
        }
        return nil
    }

    private static func header(_ file: URL) throws -> [String: Any] {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        guard let size = try handle.read(upToCount: 8), size.count == 8 else { throw LocalImageError.missing(file.path) }
        let count = size.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard count > 0, count <= 16 * 1024 * 1024,
              let data = try handle.read(upToCount: Int(count)), data.count == Int(count),
              let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LocalImageError.missing(file.path)
        }
        return result
    }
}

final class QwenImage21Control {
    let weights: QwenImage21Weights
    let inputChannels: Int
    let blocks: Int

    init(weights: QwenImage21Weights, channels: Int, dimension: Int, headDim: Int, blocks: Int) throws {
        self.weights = weights; self.blocks = blocks; inputChannels = 2 * channels + 1
        func require(_ name: String, _ shape: [Int]) throws {
            guard let value = weights.arrays[name], value.shape == shape,
                  [.float16, .bfloat16, .float32].contains(value.dtype) else {
                throw LocalImageError.unsupported("ControlNet " + name)
            }
        }
        try require("control_img_in.weight", [dimension, inputChannels])
        try require("control_img_in.bias", [dimension])
        try require("control_blocks.0.before_proj.weight", [dimension, dimension])
        try require("control_blocks.0.before_proj.bias", [dimension])
        for block in 0..<blocks {
            let p = "control_blocks.\(block)."
            for name in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "after_proj"] {
                try require(p + name + ".weight", [dimension, dimension])
            }
            for name in ["attn.norm_q.weight", "attn.norm_k.weight"] { try require(p + name, [headDim]) }
            for name in ["img_mlp.gate_layer", "img_mlp.proj"] { try require(p + name + ".weight", [3 * dimension, dimension]) }
            try require(p + "img_mlp.out.weight", [dimension, 3 * dimension])
            try require(p + "after_proj.bias", [dimension])
        }
    }

    convenience init(file: URL, transformer: QwenImage21Transformer) throws {
        let arrays = try MLX.loadArrays(url: file)
        try self.init(weights: QwenImage21Weights(arrays: arrays), channels: 64,
            dimension: transformer.heads * transformer.headDim, headDim: transformer.headDim,
            blocks: (transformer.layers + 1) / 2)
    }
}
