import Foundation

private actor PendingPrompt {
    private var continuation: CheckedContinuation<String, Error>?
    var waiting: Bool { continuation != nil }
    func generate() async throws -> String {
        try await withCheckedThrowingContinuation { continuation = $0 }
    }
    func finish(_ value: String) {
        continuation?.resume(returning: value); continuation = nil
    }
    func fail() {
        continuation?.resume(throwing: LocalReferencePromptService.PromptError.empty); continuation = nil
    }
}

@main @MainActor
struct ReferenceImagePromptTests {
    static func wait(_ condition: @MainActor () async -> Bool) async throws {
        for _ in 0..<2000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        fatalError("Timed out waiting for prompt lifecycle")
    }

    static func main() async throws {
        setbuf(stdout, nil)
        let fixture = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixture) }
        for name in ["tokenizer.json", "tokenizer_config.json", "model.safetensors"] {
            try Data("{}".utf8).write(to: fixture.appendingPathComponent(name))
        }
        try Data(#"{"model_type":"gemma4","vision_config":{}}"#.utf8)
            .write(to: fixture.appendingPathComponent("config.json"))
        for name in ["processor_config.json", "preprocessor_config.json"] {
            try Data("{}".utf8).write(to: fixture.appendingPathComponent(name))
            let models = try LocalChatModelCache.scan(in: fixture, llmTypes: ["gemma4"], vlmTypes: ["gemma4"])
            precondition(models.count == 1 && models[0].supportsImages)
            try FileManager.default.removeItem(at: fixture.appendingPathComponent(name))
        }
        let textModels = try LocalChatModelCache.scan(in: fixture, llmTypes: ["gemma4"], vlmTypes: ["gemma4"])
        precondition(textModels.count == 1 && !textModels[0].supportsImages)
        print("PASS: both processor filenames enable vision; missing processors remain text-only")
        let pending = PendingPrompt()
        let store = ImageStudioStore(promptGenerator: { _, _, _ in try await pending.generate() })
        store.promptModels = [.init(id: "vision", directory: URL(fileURLWithPath: "/tmp/vision"),
            modelType: "gemma4", isVisionModel: true, supportsImages: true, weightBytes: 0)]
        store.promptModelID = "vision"
        store.reference = Data([1]); store.prompt = "保留原有编辑指令"
        store.describeReference()
        precondition(store.busy && store.describingReference)
        try await wait { await pending.waiting }
        store.reference = Data([2])
        await pending.finish("旧图片的过期描述") // Deliberately ignores cancellation.
        try await wait { !store.busy }
        precondition(store.promptDraft.isEmpty && store.prompt == "保留原有编辑指令")
        print("PASS: changing reference cancels and rejects a late caption")

        store.describeReference()
        try await wait { await pending.waiting }
        await pending.finish("红色陶瓷杯，灰色背景，柔和侧光。")
        try await wait { !store.busy }
        precondition(!store.promptDraft.isEmpty && store.prompt == "保留原有编辑指令")
        store.usePromptDraft()
        precondition(store.prompt == "红色陶瓷杯，灰色背景，柔和侧光。" && store.promptDraft.isEmpty)
        print("PASS: draft never overwrites an existing instruction until explicitly applied")

        store.describeReference()
        try await wait { await pending.waiting }
        store.stop()
        precondition(store.stopping && store.busy)
        await pending.finish("不应使用的取消结果")
        try await wait { !store.busy }
        precondition(store.promptDraft.isEmpty && store.promptError == nil)
        store.describeReference()
        try await wait { await pending.waiting }
        await pending.fail()
        try await wait { !store.busy }
        precondition(store.promptError != nil && store.promptDraft.isEmpty)
        store.promptModelID = "missing"
        store.describeReference()
        precondition(!store.busy && store.promptError != nil)
        print("PASS: stop, failure, retry and unavailable model do not apply a draft")

        let parsed = try LocalReferencePromptService.prompt(from: .init(
            text: "<think>内部思考</think>\n\n画面描述", tokensPerSecond: 0))
        precondition(parsed == "画面描述")
        for result in [LocalChatResult(text: "<think>只有思考", tokensPerSecond: 0),
                       LocalChatResult(text: "未完成", tokensPerSecond: 0, finishReason: "length")] {
            do { _ = try LocalReferencePromptService.prompt(from: result); fatalError("Accepted incomplete result") }
            catch is LocalReferencePromptService.PromptError {}
        }
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        do {
            _ = try await LocalReferencePromptService.generate(modelID: "missing", image: Data([0]), cache: cache)
            fatalError("Accepted invalid image")
        } catch LocalReferencePromptService.PromptError.invalidImage {}
        print("PASS: reasoning is excluded; invalid images and incomplete output are rejected")

        guard CommandLine.arguments.count > 3 else { return }
        let imageURL = URL(fileURLWithPath: CommandLine.arguments[3])
        let start = Date()
        let prompt = try await LocalReferencePromptService.generate(modelID: CommandLine.arguments[2],
            image: Data(contentsOf: imageURL), cache: cache, chat: LocalChatService())
        print("PROMPT: \(prompt)")
        precondition(!prompt.isEmpty && !prompt.contains("</think>"))
        if imageURL.lastPathComponent.contains("red-mug") {
            precondition(prompt.contains("红") && prompt.contains("杯"), "Caption missed the visible red mug")
        }
        print("PASS: actual offline image-to-prompt generation in \(Date().timeIntervalSince(start)) seconds")
    }
}
