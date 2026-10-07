import Foundation

/// Resolves standard Hugging Face hub caches without changing their contents.
public enum LocalASRCache {
    public struct Selection: Sendable {
        public let directory: URL
        public let modelId: String
    }

    public static let modelIds = [
        "aufklarer/Qwen3-ASR-1.7B-MLX-4bit",
        "mlx-community/Qwen3-ASR-1.7B-bf16",
        "aufklarer/Qwen3-ASR-1.7B-MLX-8bit",
    ]

    public static func resolve(in root: URL) -> Selection? {
        // Also accept an exported model folder or a snapshot itself.
        if let id = modelId(in: root), complete(root) {
            return Selection(directory: root, modelId: id)
        }
        for hub in [root, root.appendingPathComponent("hub")] {
            for id in modelIds {
                let repository = hub.lastPathComponent == "models--" + id.replacingOccurrences(of: "/", with: "--")
                    ? hub : hub.appendingPathComponent("models--" + id.replacingOccurrences(of: "/", with: "--"))
                let snapshots = repository.appendingPathComponent("snapshots")
                var candidates: [URL] = []
                if let revision = try? String(contentsOf: repository.appendingPathComponent("refs/main"), encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                   !revision.isEmpty, revision.allSatisfy({ $0.isHexDigit }) {
                    candidates.append(snapshots.appendingPathComponent(revision))
                }
                let other = (try? FileManager.default.contentsOfDirectory(
                    at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
                candidates += other.sorted {
                    let lhs = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    let rhs = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    return lhs == rhs ? $0.lastPathComponent < $1.lastPathComponent : lhs > rhs
                }
                for directory in candidates where complete(directory) && modelId(in: directory) == id {
                    return Selection(directory: directory, modelId: id)
                }
            }
        }
        return nil
    }

    private static func complete(_ directory: URL) -> Bool {
        let required = ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
        guard required.allSatisfy({ FileManager.default.isReadableFile(atPath: directory.appendingPathComponent($0).path) }),
              HuggingFaceDownloader.weightsExist(in: directory) else { return false }
        let weights = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return weights.contains { $0.pathExtension == "safetensors" && FileManager.default.isReadableFile(atPath: $0.path) }
    }

    private static func modelId(in directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              config["model_type"] as? String == "qwen3_asr" else { return nil }
        if let thinker = config["thinker_config"] as? [String: Any],
           let text = thinker["text_config"] as? [String: Any],
           text["hidden_size"] as? Int == 2048,
           text["num_hidden_layers"] as? Int == 28,
           config["quantization"] == nil {
            return "mlx-community/Qwen3-ASR-1.7B-bf16"
        }
        if config["model_size"] as? String == "1.7B",
           let quantization = config["quantization_config"] as? [String: Any],
           quantization["group_size"] as? Int == 64,
           let bits = quantization["bits"] as? Int, [4, 8].contains(bits) {
            return "aufklarer/Qwen3-ASR-1.7B-MLX-\(bits)bit"
        }
        return nil
    }
}
