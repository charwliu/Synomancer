import Foundation

@main
struct LocalChatReasoningSmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let id = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit"
        let service = LocalChatService()
        let history: [LocalChatMessage] = [
            .init(role: "user", content: "The passphrase is REDWOOD-472."),
            .init(role: "assistant", content: "I will remember REDWOOD-472."),
            .init(role: "user", content: "Reply only with the exact passphrase I gave you.")
        ]
        for mtp in [false, true] {
            let (snapshots, continuation) = AsyncStream<String>.makeStream()
            let generation = Task {
                defer { continuation.finish() }
                return try await service.generate(modelID: id, messages: history, cache: cache,
                    temperature: 0, maxTokens: 256, useMTP: mtp, frameReasoning: true) { text, _ in
                        continuation.yield(text)
                    }
            }
            var sawThinking = false
            var count = 0
            for await text in snapshots {
                let parsed = AssistantMessageContent(text, streaming: true)
                precondition(parsed.reasoning != nil, "Prefilled thinking was exposed as the answer: \(text)")
                precondition(!parsed.answer.contains("</think>"))
                precondition(!(parsed.reasoning ?? "").contains("</think>"))
                if parsed.isThinking { sawThinking = true; precondition(parsed.answer.isEmpty) }
                count += 1
            }
            let result = try await generation.value
            let parsed = AssistantMessageContent(result.text)
            precondition(sawThinking && count > 0 && !parsed.isThinking)
            precondition(parsed.answer == "REDWOOD-472", "Unexpected final answer: \(parsed.answer)")
            if mtp { precondition((result.proposedDraftTokens ?? 0) > 0) }
            print("PASS: real Qwen \(mtp ? "MTP" : "ordinary") streaming thoughts stay separate; final copy/speech text=\(parsed.answer); tokens=\(result.completionTokens)")
        }
        let unfinished = try await service.generate(modelID: id, messages: history, cache: cache,
            temperature: 0, maxTokens: 4, useMTP: false, frameReasoning: true) { _, _ in }
        let parsed = AssistantMessageContent(unfinished.text)
        precondition(parsed.isThinking && parsed.answer.isEmpty)
        print("PASS: a length-limited thinking-only response cannot be copied/spoken as an answer")
    }
}
