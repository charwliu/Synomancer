import Foundation

@main
struct LocalChatSmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let service = LocalChatService()
        let models = try await service.models(cache: cache)
        precondition(!models.isEmpty)
        precondition(!models.contains { $0.modelType.contains("tts") || $0.modelType.contains("asr") })
        print("PASS: supported complete local models: \(models.map(\.id).joined(separator: ", "))")
        let id = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : models[0].id
        let start = Date()
        let result = try await service.generate(modelID: id, messages: [
            .init(role: "system", content: "You are a helpful assistant. Follow the user's instructions."),
            .init(role: "user", content: "The passphrase is REDWOOD-472."),
            .init(role: "assistant", content: "I will remember the passphrase REDWOOD-472."),
            .init(role: "user", content: "Reply only with the exact passphrase I gave you."),
        ], cache: cache, temperature: 0, maxTokens: 64) { _, _ in }
        print("OUTPUT: \(result.text)")
        precondition(result.text.contains("REDWOOD-472"))
        print("PASS: real offline MLX chat preserves multi-turn history; time=\(Date().timeIntervalSince(start))s speed=\(result.tokensPerSecond) tokens/s")

        if models.first(where: { $0.id == id })!.isVisionModel && !models.first(where: { $0.id == id })!.supportsImages {
            do {
                _ = try await service.generate(modelID: id, messages: [.init(role: "user", content: "Describe this image.", image: Data([1]))],
                                               cache: cache) { _, _ in }
                fatalError("Unvalidated vision input was accepted")
            } catch LocalChatService.ChatError.imagesUnsupported {
                print("PASS: unvalidated vision input is rejected before inference")
            }
        }

        let (started, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let old = Task {
            try await service.generate(modelID: id, messages: [
                .init(role: "user", content: "Write a detailed story of at least 2000 words about exploring the ocean."),
            ], cache: cache, temperature: 0, maxTokens: 2048) { _, _ in continuation.yield(()) }
        }
        var iterator = started.makeAsyncIterator()
        _ = await iterator.next()
        old.cancel()
        let replacement = Task {
            try await service.generate(modelID: id, messages: [.init(role: "user", content: "Reply only with OK.")],
                                       cache: cache, temperature: 0, maxTokens: 16) { _, _ in }
        }
        do {
            _ = try await old.value
            fatalError("Cancelled MLX generation returned success")
        } catch is CancellationError { print("PASS: local MLX token generation cooperates with cancellation") }
        let next = try await replacement.value
        precondition(next.text.uppercased().contains("OK"))
        print("PASS: immediate replacement reuses cached model after cancellation: \(next.text)")
        continuation.finish()
    }
}
