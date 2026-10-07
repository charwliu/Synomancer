import Foundation

@main
struct LocalChatMoESmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let id = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "mlx-community/Qwen3.5-122B-A10B-oQ4-MTP"
        let service = LocalChatService()
        let models = try await service.models(cache: cache)
        guard let selected = models.first(where: { $0.id == id }) else { fatalError("Target cache not discovered") }
        print("MODEL: \(id) vision=\(selected.isVisionModel) weights=\(selected.weightBytes) type=\(selected.modelType)")
        for framed in [false, true] {
            let start = Date()
            let result = try await service.generate(modelID: id,
                messages: [.init(role: "user", content: "Reply with only Hello.")], cache: cache,
                temperature: 0, maxTokens: 512, useMTP: true, frameReasoning: framed) { text, count in
                    if count <= 8 { print("STREAM tokens=\(count) text=\(String(reflecting: text))") }
                }
            print("RESULT: framed=\(framed) elapsed=\(Date().timeIntervalSince(start))s tokens=\(result.completionTokens) finish=\(result.finishReason) text=\(String(reflecting: result.text))")
            let display = AssistantMessageContent(result.text)
            print("DISPLAY: thinking=\(display.isThinking) reasoning=\(String(reflecting: display.reasoning)) answer=\(String(reflecting: display.answer))")
            precondition(result.finishReason == "stop" && result.completionTokens > 0)
            precondition(!display.isThinking && display.answer.contains("Hello"))
            precondition(result.proposedDraftTokens == nil, "Unsupported MoE MTP must use ordinary generation")
            if framed {
                precondition(display.reasoning != nil && !display.answer.contains("</think>"))
                print("PASS: real 122B MoE generates a complete App answer; thinking/copy/speech are separated")
            } else {
                precondition(!result.text.isEmpty)
                print("PASS: real 122B MoE raw generation works with MTP parameters excluded from the main model")
            }
        }
    }
}
