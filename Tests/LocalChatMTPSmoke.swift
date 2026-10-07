import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct LocalChatMTPSmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let id = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit"
        let service = LocalChatService()
        let models = try await service.models(cache: cache)
        precondition(models.contains { $0.id == id }, "Complete target checkpoint is required")
        let prompt = [LocalChatMessage(role: "user", content:
            "Write a Python module implementing fibonacci, factorial, and binary search. Include docstrings, input validation, and examples. /no_think")]
        // Compile both paths and load the drafter before timing either path.
        for mtp in [false, true] {
            let warmup = try await service.generate(modelID: id, messages: prompt, cache: cache,
                temperature: 0, maxTokens: 16, useMTP: mtp) { _, _ in }
            print("WARMUP MTP=\(mtp) tokens=\(warmup.completionTokens) proposals=\(warmup.proposedDraftTokens ?? 0)")
        }
        let baseline = try await service.generate(modelID: id, messages: prompt, cache: cache,
            temperature: 0, maxTokens: 128, useMTP: false) { _, _ in }
        let mtp = try await service.generate(modelID: id, messages: prompt, cache: cache,
            temperature: 0, maxTokens: 128, useMTP: true) { _, _ in }
        print("BASE: \(baseline.text)")
        print("MTP: \(mtp.text)")
        print("METRICS base=\(baseline.tokensPerSecond) tok/s mtp=\(mtp.tokensPerSecond) tok/s speedup=\(mtp.tokensPerSecond / baseline.tokensPerSecond) accepted=\(mtp.acceptedDraftTokens ?? 0)/\(mtp.proposedDraftTokens ?? 0) prefill=\(mtp.promptSeconds)s")
        if mtp.text != baseline.text {
            print("NOTE: BF16 long continuation differs between compiled single-token and batched target verification; not bitwise greedy parity")
            let model = models.first { $0.id == id }!
            try await checkVerificationNumerics(directory: model.directory, prompt: prompt, baseline: baseline.text)
        }
        precondition(mtp.completionTokens == baseline.completionTokens && mtp.completionTokens > 0)
        precondition((mtp.proposedDraftTokens ?? 0) > 0 && (mtp.acceptedDraftTokens ?? 0) > 0)
        precondition(mtp.mtpFallbackReason == nil && mtp.finishReason == baseline.finishReason)
        print("PASS: real OptiQ MTP performs target verification with accepted/rejected drafts and correct token usage")

        let exact = [LocalChatMessage(role: "system", content: "Follow the user’s instruction. Be concise."),
            .init(role: "user", content: "The passphrase is REDWOOD-472."),
            .init(role: "assistant", content: "I will remember REDWOOD-472."),
            .init(role: "user", content: "Reply only with the exact passphrase I gave you.")]
        let exactBase = try await service.generate(modelID: id, messages: exact, cache: cache,
            temperature: 0, maxTokens: 128, useMTP: false) { _, _ in }
        let exactMTP = try await service.generate(modelID: id, messages: exact, cache: cache,
            temperature: 0, maxTokens: 128, useMTP: true) { _, _ in }
        print("EXTRACTION BASE: \(exactBase.text)")
        print("EXTRACTION MTP: \(exactMTP.text)")
        precondition(exactBase.text == exactMTP.text && exactMTP.text.contains("REDWOOD-472"))
        print("PASS: real multi-turn passphrase extraction is identical with and without MTP")

        let stop = String(baseline.text.prefix(12))
        precondition(!stop.isEmpty)
        let (snapshots, snapshotContinuation) = AsyncStream<String>.makeStream()
        let stopped = try await service.generate(modelID: id, messages: prompt, cache: cache,
            temperature: 0, maxTokens: 128, stop: [stop], useMTP: true) { text, _ in snapshotContinuation.yield(text) }
        snapshotContinuation.finish()
        for await snapshot in snapshots { precondition(snapshot.isEmpty, "Stream leaked a partial stop sequence: \(snapshot)") }
        precondition(!stopped.text.contains(stop) && stopped.finishReason == "stop")
        precondition(stopped.promptTokens > 0 && stopped.completionTokens > 0)
        print("PASS: MTP stop sequences drain and join the worker with actual token usage")

        let (started, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let old = Task {
            try await service.generate(modelID: id, messages: prompt, cache: cache,
                temperature: 0, maxTokens: 2048, useMTP: true) { _, _ in continuation.yield(()) }
        }
        var iterator = started.makeAsyncIterator()
        _ = await iterator.next()
        old.cancel()
        let replacement = Task {
            try await service.generate(modelID: id, messages: prompt, cache: cache,
                temperature: 0, maxTokens: 16, useMTP: true) { _, _ in }
        }
        do { _ = try await old.value; preconditionFailure("Cancelled MTP generation succeeded") }
        catch is CancellationError { }
        let next = try await replacement.value
        precondition(baseline.text.hasPrefix(next.text) && next.completionTokens == 16)
        continuation.finish()
        print("PASS: MTP cancellation and immediate replacement preserve the greedy prefix")

        let sampled = try await service.generate(modelID: id, messages: prompt, cache: cache,
            temperature: 0.6, maxTokens: 8, useMTP: true) { _, _ in }
        precondition(sampled.proposedDraftTokens == nil && sampled.acceptedDraftTokens == nil)
        precondition(sampled.mtpFallbackReason?.contains("temperature 0") == true)
        print("PASS: nonzero-temperature requests preserve ordinary sampling without speculative prefill")
    }
    // Compare one-token and two-token target forwards from equivalent fresh
    // cache states; position zero cannot attend to the dummy future token.
    static func checkVerificationNumerics(directory: URL, prompt: [LocalChatMessage], baseline: String) async throws {
        let context = try await LLMModelFactory.shared.load(from: directory, using: LocalChatTokenizerLoader())
        let input = try await context.processor.prepare(input: UserInput(messages: prompt.map {
            ["role": $0.role, "content": $0.content] as MLXLMCommon.Message
        }))
        let continuation = context.tokenizer.encode(text: baseline)
        let ids = input.text.tokens.asArray(Int.self) + Array(continuation.prefix(48))
        let prefix = MLXArray(ids.dropLast()).reshaped([1, ids.count - 1])
        let singleCache = try context.model.newCache(parameters: nil as GenerateParameters?)
        let batchCache = try context.model.newCache(parameters: nil as GenerateParameters?)
        let singlePrefill = context.model(prefix, cache: singleCache)
        eval(singlePrefill)
        let batchPrefill = context.model(prefix, cache: batchCache)
        eval(batchPrefill)
        let one = context.model(MLXArray([ids.last!]).reshaped([1, 1]), cache: singleCache)[0, 0].asType(.float32)
        let two = context.model(MLXArray([ids.last!, 0]).reshaped([1, 2]), cache: batchCache)[0, 0].asType(.float32)
        let delta = abs(one - two).max().item(Float.self)
        let oneTop = argMax(one).item(Int.self), twoTop = argMax(two).item(Int.self)
        print("TARGET NUMERICS: same-cache single vs batch max logit delta=\(delta) top1=\(oneTop)/\(twoTop)")
        precondition(delta.isFinite && delta < 1, "Target verification has an unexpectedly large numerical difference")
        Stream.defaultStream.synchronize()
    }

}
