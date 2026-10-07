import Foundation

@main
struct LocalChatCacheTests {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("synomancer-chat-cache-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let hub = root.appendingPathComponent("hub")
        let repo = hub.appendingPathComponent("models--mlx-community--test-chat")
        let snapshot = repo.appendingPathComponent("snapshots/abc123")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: repo.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try "abc123".write(to: repo.appendingPathComponent("refs/main"), atomically: true, encoding: .utf8)
        func write(_ name: String, _ data: String) throws {
            try data.write(to: snapshot.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        try write("config.json", "{\"model_type\":\"qwen2\"}")
        try write("tokenizer.json", "{}")
        try write("tokenizer_config.json", "{}")
        let blob = hub.appendingPathComponent("blob")
        try Data([1, 2, 3]).write(to: blob)
        let weights = snapshot.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: blob)
        func scan(_ directory: URL) throws -> [LocalChatModel] {
            try LocalChatModelCache.scan(in: directory, llmTypes: ["qwen2", "qwen3_5"], vlmTypes: ["qwen2_vl", "qwen3_5"])
        }
        for directory in [root, hub, repo] {
            let models = try scan(directory)
            precondition(models.count == 1 && models[0].id == "mlx-community/test-chat" && models[0].weightBytes == 3)
        }
        let direct = try scan(snapshot)
        precondition(direct.count == 1)
        try blob.delete()
        let empty0 = try scan(hub)
        precondition(empty0.isEmpty, "Dangling weight symlinks must not be listed")
        try Data([1]).write(to: blob)
        try write("model.safetensors.index.json", "{\"weight_map\":{\"a\":\"model.safetensors\",\"b\":\"missing.safetensors\"}}")
        let empty1 = try scan(hub)
        precondition(empty1.isEmpty, "Missing shards must not be listed")
        try snapshot.appendingPathComponent("model.safetensors.index.json").delete()
        try write("config.json", "{\"model_type\":\"qwen3_tts\"}")
        let empty2 = try scan(hub)
        precondition(empty2.isEmpty)
        try write("config.json", "{\"model_type\":\"qwen2_vl\"}")
        let empty3 = try scan(hub)
        precondition(empty3.isEmpty, "Vision models need a local processor config")
        try write("preprocessor_config.json", "{}")
        let vision = try scan(hub)
        precondition(vision.first?.isVisionModel == true && vision.first?.supportsImages == false)
        try snapshot.appendingPathComponent("preprocessor_config.json").delete()
        try write("config.json", "{\"model_type\":\"qwen3_5\",\"text_config\":{}}")
        let textOnly = try scan(hub)
        precondition(textOnly.count == 1 && !textOnly[0].isVisionModel && !textOnly[0].supportsImages)
        try write("config.json", "{\"model_type\":\"qwen3_5\",\"text_config\":{},\"vision_config\":{}}")
        let withoutProcessor = try scan(hub)
        precondition(withoutProcessor.count == 1 && !withoutProcessor[0].isVisionModel)
        try write("preprocessor_config.json", "{}")
        let combined = try scan(hub)
        precondition(combined.count == 1 && combined[0].isVisionModel)
        try write("config.json", "{\"model_type\":\"qwen3_5\",\"vision_config\":{},\"language_model_only\":true}")
        let explicitText = try scan(hub)
        precondition(explicitText.count == 1 && !explicitText[0].isVisionModel)
        try write("config.json", "{\"model_type\":\"qwen3_5\",\"vision_config\":null}")
        let nullVision = try scan(hub)
        precondition(nullVision.count == 1 && !nullVision[0].isVisionModel)
        let candidates = try LocalChatModelCache.scan(in: hub)
        precondition(candidates.count == 1 && candidates[0].modelType == "qwen3_5")
        let alternate = repo.appendingPathComponent("snapshots/def456")
        try FileManager.default.createDirectory(at: alternate, withIntermediateDirectories: true)
        for file in ["tokenizer.json", "tokenizer_config.json", "model.safetensors"] {
            try FileManager.default.copyItem(at: snapshot.appendingPathComponent(file), to: alternate.appendingPathComponent(file))
        }
        try Data("{\"model_type\":\"qwen2\"}".utf8).write(to: alternate.appendingPathComponent("config.json"))
        let all = try LocalChatModelCache.scan(in: hub, allSnapshots: true)
        precondition(Set(all.map(\.modelType)) == ["qwen2", "qwen3_5"])
        print("PASS: HF layouts, incomplete caches, all-snapshot discovery and Qwen3.5 text/vision routing")
    }
}

private extension URL {
    func delete() throws { try FileManager.default.removeItem(at: self) }
}
