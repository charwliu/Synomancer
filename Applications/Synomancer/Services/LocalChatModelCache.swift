import Foundation

struct LocalChatModel: Sendable, Equatable, Identifiable {
    let id: String
    let directory: URL
    let modelType: String
    let isVisionModel: Bool
    let supportsImages: Bool
    let weightBytes: UInt64
}

/// Reads complete HF snapshots and exported model directories without downloads.
enum LocalChatModelCache {
    static func scan(in root: URL, llmTypes: Set<String>? = nil, vlmTypes: Set<String>? = nil, allSnapshots: Bool = false) throws -> [LocalChatModel] {
        var models: [String: LocalChatModel] = [:]
        func add(_ directory: URL, id: String) {
            let key = allSnapshots ? directory.standardizedFileURL.path : id
            if models[key] == nil, let model = inspect(directory, id: id, llmTypes: llmTypes, vlmTypes: vlmTypes) {
                models[key] = model
            }
        }
        let repository = root.deletingLastPathComponent().deletingLastPathComponent()
        let directID = root.deletingLastPathComponent().lastPathComponent == "snapshots" && repository.lastPathComponent.hasPrefix("models--")
            ? String(repository.lastPathComponent.dropFirst(8)).replacingOccurrences(of: "--", with: "/")
            : root.lastPathComponent
        add(root, id: directID)
        for hub in [root, root.appendingPathComponent("hub")] {
            let repositories: [URL]
            if hub.lastPathComponent.hasPrefix("models--") { repositories = [hub] }
            else { repositories = (try? FileManager.default.contentsOfDirectory(at: hub, includingPropertiesForKeys: nil)) ?? [] }
            for repository in repositories where repository.lastPathComponent.hasPrefix("models--") {
                let id = String(repository.lastPathComponent.dropFirst(8)).replacingOccurrences(of: "--", with: "/")
                let snapshots = repository.appendingPathComponent("snapshots")
                var candidates: [URL] = []
                if let revision = try? String(contentsOf: repository.appendingPathComponent("refs/main"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
                   !revision.isEmpty, revision.allSatisfy(\.isHexDigit) {
                    candidates.append(snapshots.appendingPathComponent(revision))
                }
                let others = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
                candidates += others.sorted {
                    let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    return a == b ? $0.lastPathComponent < $1.lastPathComponent : a > b
                }
                for candidate in candidates { add(candidate, id: id) }
            }
        }
        // Prefer smaller cached models for low latency, without loading weights.
        return models.values.sorted { $0.weightBytes == $1.weightBytes ? $0.id < $1.id : $0.weightBytes < $1.weightBytes }
    }

    private static func inspect(_ directory: URL, id: String, llmTypes: Set<String>? = nil, vlmTypes: Set<String>? = nil) -> LocalChatModel? {
        if let descriptor = try? Qwen4GGUFDescriptor.read(directory),
           readableFile(directory.appendingPathComponent("config.json")),
           readableFile(directory.appendingPathComponent("tokenizer.json")),
           readableFile(directory.appendingPathComponent("tokenizer_config.json")),
           readableFile(directory.appendingPathComponent("preprocessor_config.json")),
           descriptor.shards.allSatisfy({ readableFile(directory.appendingPathComponent($0)) }) {
            let images = descriptor.projector.map { readableFile(directory.appendingPathComponent($0)) } ?? false
            let bytes = descriptor.shards.reduce(UInt64(0)) { total, path in
                total + ((try? FileManager.default.attributesOfItem(atPath: directory.appendingPathComponent(path).resolvingSymlinksInPath().path)[.size] as? NSNumber)?.uint64Value ?? 0)
            }
            return LocalChatModel(id: id, directory: directory, modelType: "qwen4_exp_gguf", isVisionModel: images, supportsImages: images, weightBytes: bytes)
        }
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = config["model_type"] as? String,
              llmTypes == nil || llmTypes!.contains(type) || vlmTypes?.contains(type) == true,
              readableFile(directory.appendingPathComponent("tokenizer.json")),
              readableFile(directory.appendingPathComponent("tokenizer_config.json")) else { return nil }
        let hasVisionAssets = config["vision_config"] as? [String: Any] != nil
            && config["language_model_only"] as? Bool != true
            && readableFile(directory.appendingPathComponent("preprocessor_config.json"))
        // Shared model_type keys (Qwen3.5, Gemma3…) can also be text-only checkpoints.
        let images = vlmTypes?.contains(type) == true
            && (llmTypes?.contains(type) != true || hasVisionAssets)
        if images && !readableFile(directory.appendingPathComponent("preprocessor_config.json")) { return nil }
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        let weights: [URL]
        if FileManager.default.fileExists(atPath: index.path) {
            guard let data = try? Data(contentsOf: index),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let map = json["weight_map"] as? [String: String], !map.isEmpty,
                  map.values.allSatisfy({ !$0.contains("/") && $0.hasSuffix(".safetensors") }) else { return nil }
            weights = Set(map.values).map { directory.appendingPathComponent($0) }
        } else {
            weights = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "safetensors" }
        }
        guard !weights.isEmpty, weights.allSatisfy(readableFile) else { return nil }
        let size = weights.reduce(UInt64(0)) { total, url in
            total + ((try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)[.size] as? NSNumber)?.uint64Value ?? 0)
        }
        return LocalChatModel(id: id, directory: directory, modelType: type, isVisionModel: images, supportsImages: images && type != "qwen2_vl", weightBytes: size)
    }

    private static func readableFile(_ url: URL) -> Bool {
        let target = url.resolvingSymlinksInPath()
        return FileManager.default.isReadableFile(atPath: target.path)
            && ((try? target.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])).map { $0.isRegularFile == true && ($0.fileSize ?? 0) > 0 } ?? false)
    }
}
