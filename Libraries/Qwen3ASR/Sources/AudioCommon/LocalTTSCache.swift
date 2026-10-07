import Foundation

public enum LocalTTSCache {
    public struct Selection: Sendable {
        public let model: URL
        public let tokenizer: URL
    }

    public static func resolve(in root: URL) -> Selection? {
        var models = [root]
        for hub in [root, root.appendingPathComponent("hub")] {
            for id in ["mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"] {
                let name = "models--" + id.replacingOccurrences(of: "/", with: "--")
                let repo = hub.lastPathComponent == name ? hub : hub.appendingPathComponent(name)
                models += snapshots(in: repo)
            }
        }
        for model in models where validModel(model) {
            var tokenizers = [model.appendingPathComponent("speech_tokenizer")]
            for hub in [root, root.appendingPathComponent("hub")] {
                tokenizers += snapshots(in: hub.appendingPathComponent("models--Qwen--Qwen3-TTS-Tokenizer-12Hz"))
            }
            if let tokenizer = tokenizers.first(where: { HuggingFaceDownloader.weightsExist(in: $0) }) {
                return Selection(model: model, tokenizer: tokenizer)
            }
        }
        return nil
    }

    private static func validModel(_ directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              config["model_type"] as? String == "qwen3_tts",
              config["tts_model_type"] as? String == "custom_voice",
              let talker = config["talker_config"] as? [String: Any],
              talker["hidden_size"] as? Int == 2048,
              FileManager.default.isReadableFile(atPath: directory.appendingPathComponent("vocab.json").path),
              HuggingFaceDownloader.weightsExist(in: directory) else { return false }
        return true
    }

    private static func snapshots(in repository: URL) -> [URL] {
        let snapshots = repository.appendingPathComponent("snapshots")
        var result: [URL] = []
        if let ref = try? String(contentsOf: repository.appendingPathComponent("refs/main"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !ref.isEmpty, ref.allSatisfy({ $0.isHexDigit }) {
            result.append(snapshots.appendingPathComponent(ref))
        }
        result += ((try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil)) ?? [])
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return result
    }
}
