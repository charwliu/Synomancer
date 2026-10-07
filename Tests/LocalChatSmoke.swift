import Foundation
import Synchronization

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
        let benchmarkPrefix = ProcessInfo.processInfo.environment["SYNOMANCER_BENCH_PREFIX"] == "1"
        if benchmarkPrefix || ProcessInfo.processInfo.environment["SYNOMANCER_BENCH_PREFILL"] == "1" {
            _ = try await service.generate(modelID: id, messages: [.init(role: "user", content: "Hi")],
                cache: cache, temperature: 0, maxTokens: 1, useMTP: false,
                reusePromptCache: false) { _, _ in }
            let messages: [LocalChatMessage] = [
                .init(role: "system", content: "Follow instructions. Answer briefly."),
                .init(role: "user", content: String(repeating: "Reference note: the passphrase is REDWOOD-472. Preserve it exactly.\n", count: 96)
                    + "Reply only with the exact passphrase."),
            ]
            // ABBA order reduces warm-up and thermal ordering bias.
            for step in benchmarkPrefix ? [512, 512, 512] : [512, 1024, 1024, 512] {
                let start = ContinuousClock.now
                let first = Mutex<Double?>(nil)
                let result = try await service.generate(modelID: id, messages: messages,
                    cache: cache, temperature: 0, maxTokens: 128, useMTP: false,
                    reusePromptCache: benchmarkPrefix, prefillStepSize: step) { _, _ in
                    first.withLock { value in
                        if value == nil {
                            let elapsed = start.duration(to: .now).components
                            value = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                        }
                    }
                }
                let answer = AssistantMessageContent(result.text).answer
                print("PREFILL prefix=\(benchmarkPrefix) cached=\(result.cachedPromptTokens) step=\(step) tokens=\(result.promptTokens) first=\(first.withLock { $0! }) prefill=\(result.promptSeconds) answer=\(answer.debugDescription)")
                precondition(answer == "REDWOOD-472")
            }
            return
        }
        let updates = Mutex<[(String, Int)]>([])
        let start = Date()
        let result = try await service.generate(modelID: id, messages: [
            .init(role: "system", content: "You are a helpful assistant. Follow the user's instructions."),
            .init(role: "user", content: "The passphrase is REDWOOD-472."),
            .init(role: "assistant", content: "I will remember the passphrase REDWOOD-472."),
            .init(role: "user", content: "Reply only with the exact passphrase I gave you."),
        ], cache: cache, temperature: 0, maxTokens: 64) { text, count in
            updates.withLock { $0.append((text, count)) }
        }
        precondition(result.tokensPerSecond.isFinite && result.tokensPerSecond > 0)
        precondition(result.completionTokens > 0 && result.generationSeconds > 0)
        precondition(updates.withLock { $0.last?.0 == result.text && $0.last?.1 == result.completionTokens })
        print("PASS: final stream update is complete; measured throughput and token counts are valid")
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

        if ["qwen3_5", "qwen3_5_text", "gemma4", "gemma4_text"].contains(models.first { $0.id == id }!.modelType) {
            // Compare the same second-turn request with and without a prompt
            // checkpoint. The long shared context makes prefill savings measurable.
            let contextText = String(repeating: "Reference note: the passphrase is REDWOOD-472. Preserve it exactly.\n", count: 64)
            let firstMessages: [LocalChatMessage] = [
                .init(role: "system", content: "Follow instructions. Answer briefly."),
                .init(role: "user", content: contextText + "Reply only with OK."),
            ]
            await service.clearPromptCache()
            let first = try await service.generate(modelID: id, messages: firstMessages,
                cache: cache, temperature: 0, maxTokens: 32, useMTP: false) { _, _ in }
            let continued = firstMessages + [
                .init(role: "assistant", content: first.text),
                .init(role: "user", content: "What is the passphrase? Reply with the passphrase only."),
            ]
            func measured(_ messages: [LocalChatMessage], reuseCache: Bool = true, limit: Int = 128) async throws -> (LocalChatResult, Double) {
                let start = ContinuousClock.now
                let latency = Mutex<Double?>(nil)
                let response = try await service.generate(modelID: id, messages: messages,
                    cache: cache, temperature: 0, maxTokens: limit, useMTP: false, reusePromptCache: reuseCache) { _, _ in
                    latency.withLock { value in
                        if value == nil {
                            let elapsed = start.duration(to: .now).components
                            value = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                        }
                    }
                }
                return (response, latency.withLock { $0! })
            }
            let (warm, warmFirstToken) = try await measured(continued)
            let (warmRepeat, _) = try await measured(continued)
            precondition(warmRepeat.text == warm.text, "Same checkpoint changed greedy output")
            await service.clearPromptCache()
            let (cold, coldFirstToken) = try await measured(continued, reuseCache: false)
            print("CACHE CHECK warm=\(warm.cachedPromptTokens) cold=\(cold.cachedPromptTokens) parity=\(warm.text == cold.text)")
            precondition(warm.cachedPromptTokens > 0, "Expected an exact-prefix cache hit")
            precondition(cold.cachedPromptTokens == 0)
            print("ANSWER CHECK warm=\(AssistantMessageContent(warm.text).answer.debugDescription) warmFinish=\(warm.finishReason) cold=\(AssistantMessageContent(cold.text).answer.debugDescription) coldFinish=\(cold.finishReason)")
            precondition(AssistantMessageContent(warm.text).answer == "REDWOOD-472")
            precondition(AssistantMessageContent(cold.text).answer == "REDWOOD-472")
            print("PASS: checkpoint repeat is exact and warm/cold answers match ground truth; cached=\(warm.cachedPromptTokens)/\(warm.promptTokens) tokens; first-token warm=\(warmFirstToken)s cold=\(coldFirstToken)s; prefill warm=\(warm.promptSeconds)s cold=\(cold.promptSeconds)s")
            _ = try await measured(continued) // Rebuild a checkpoint before invalidation.
            var edited = continued
            edited[0] = .init(role: "system", content: "You are a different assistant. Reply briefly.")
            let (changed, _) = try await measured(edited)
            precondition(changed.cachedPromptTokens == 0, "Edited history reused stale state")
            print("PASS: changing the system prompt invalidates the checkpoint")
            let (cacheStarted, cacheContinuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
            let cancelled = Task {
                try await service.generate(modelID: id, messages: continued, cache: cache,
                    temperature: 0, maxTokens: 2048, useMTP: false) { _, _ in cacheContinuation.yield(()) }
            }
            var cacheIterator = cacheStarted.makeAsyncIterator()
            _ = await cacheIterator.next()
            cancelled.cancel()
            do {
                _ = try await cancelled.value
                fatalError("Cancelled cached request returned success")
            } catch is CancellationError {}
            let (retried, _) = try await measured(continued)
            precondition(retried.cachedPromptTokens > 0 && retried.text == warm.text)
            cacheContinuation.finish()
            print("PASS: cancellation leaves an independent checkpoint; retry matches the earlier cached output")
            let extended = continued + [
                .init(role: "assistant", content: retried.text),
                .init(role: "user", content: String(repeating: "Keep the passphrase unchanged.\n", count: 64) + "Reply with the passphrase only."),
            ]
            let (extendedWarm, _) = try await measured(extended, limit: 128)
            let (extendedRetry, _) = try await measured(extended, limit: 128)
            let (extendedCold, _) = try await measured(extended, reuseCache: false, limit: 128)
            // Different prefill batch boundaries can change near-tied greedy
            // reasoning tokens. Check the completed answer against ground truth,
            // while a retry from the same checkpoint must remain token-identical.
            precondition(extendedWarm.cachedPromptTokens > 0)
            precondition(extendedRetry.cachedPromptTokens > extendedWarm.cachedPromptTokens)
            precondition(extendedRetry.text == extendedWarm.text)
            precondition(AssistantMessageContent(extendedWarm.text).answer == "REDWOOD-472")
            precondition(AssistantMessageContent(extendedCold.text).answer == "REDWOOD-472")
            print("PASS: extended checkpoint retry is exact; warm/cold final answers match ground truth; rawParity=\(extendedWarm.text == extendedCold.text)")


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
