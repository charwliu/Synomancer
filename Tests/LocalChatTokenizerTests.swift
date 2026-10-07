import Foundation
import MLXLMCommon
import Tokenizers

@main
struct LocalChatTokenizerTests {
    static func main() async throws {
        setbuf(stdout, nil)
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let service = LocalChatService()
        let models = try await service.models(cache: cache)
        guard let pretrained = models.first(where: { $0.id == "mlx-community/gemma-3-27b-pt-bf16" }) else {
            fatalError("The complete Gemma 3 PT cache is required for this regression")
        }
        do {
            _ = try await service.generate(modelID: pretrained.id,
                messages: [.init(role: "user", content: "Hello")], cache: cache,
                maxTokens: 1, onUpdate: { _, _ in preconditionFailure("PT chat must not start generation") })
            preconditionFailure("Missing PT chat template must fail before weights are loaded")
        } catch LocalChatTokenizerLoader.ChatTemplateError.missing(let id, let isPT) {
            precondition(id == pretrained.id && isPT)
        }
        print("PASS: actual cached Gemma 3 PT rejected before loading weights with actionable model-specific error")

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("synomancer-tokenizer-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("tokenizer.json"),
            withDestinationURL: pretrained.directory.appendingPathComponent("tokenizer.json"))
        let original = try Data(contentsOf: pretrained.directory.appendingPathComponent("tokenizer_config.json"))
        let template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
        let messages: [[String: any Sendable]] = [["role": "user", "content": "你好，Gemma"]]
        var baseline: [Int]?
        for format in ["inline", "jinja", "json", "named"] {
            for name in ["chat_template.jinja", "chat_template.json"] {
                let file = root.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: file.path) { try FileManager.default.removeItem(at: file) }
            }
            var config = try JSONSerialization.jsonObject(with: original) as! [String: Any]
            if format == "inline" { config["chat_template"] = template }
            if format == "named" { config["chat_template"] = [["name": "default", "template": template]] }
            try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("tokenizer_config.json"))
            if format == "jinja" {
                try template.write(to: root.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
            } else if format == "json" {
                try JSONSerialization.data(withJSONObject: ["chat_template": template]).write(to: root.appendingPathComponent("chat_template.json"))
            }
            let model = LocalChatModel(id: "test/gemma-it", directory: root, modelType: "gemma3",
                isVisionModel: true, supportsImages: true, weightBytes: 0)
            let loader = try await LocalChatTokenizerLoader.validated(for: model)
            // The validated loader must remain usable without re-reading the files.
            try FileManager.default.removeItem(at: root.appendingPathComponent("tokenizer_config.json"))
            let tokenizer = try await loader.load(from: root)
            let tokens = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
            precondition(!tokens.isEmpty)
            precondition(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true) == "你好，Gemma")
            if let baseline { precondition(tokens == baseline) } else { baseline = tokens }
            print("PASS: \(format) local chat template produces identical tokens; preflight tokenizer is reused")
        }
        try original.write(to: root.appendingPathComponent("tokenizer_config.json"))
        let missing = LocalChatModel(id: "test/gemma-it", directory: root, modelType: "gemma3",
            isVisionModel: false, supportsImages: false, weightBytes: 0)
        do {
            _ = try await LocalChatTokenizerLoader.validated(for: missing)
            preconditionFailure("Incomplete IT cache must be reported")
        } catch LocalChatTokenizerLoader.ChatTemplateError.missing(let id, let isPT) {
            precondition(id == missing.id && !isPT)
        }
        print("PASS: incomplete IT cache distinguished from PT model; original HF cache remains read-only")
    }
}
