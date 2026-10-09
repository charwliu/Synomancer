import Foundation
import MLX

@main
struct Qwen4GGUFSmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        let args = CommandLine.arguments
        if args.count == 2, args[1] == "--kernels" { try GGUFKernelTests.run(); return }
        if args.count == 4, args[1] == "--integration" { try await integration(directory: args[2], image: args[3]); return }
        guard args.count == 3 else { fatalError("usage: Qwen4GGUFSmoke first-shard.gguf input.json") }
        MLX.Memory.cacheLimit = 1 << 30
        let ticket = WiredMemoryTicket(size: 105 * 1024 * 1024 * 1024, policy: WiredMaxPolicy())
        try await ticket.withWiredLimit {
            let start = Date()
            let model = try Qwen4GGUFModel(firstShard: URL(fileURLWithPath: args[1]))
            print("Loaded in", Date().timeIntervalSince(start)); fflush(stdout)
            let sidecar = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_MTP"]
            let draft = try sidecar.map { try Qwen4GGUFMTP(sidecar: URL(fileURLWithPath: $0)) }
            let extraFixture = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_EXTRA_FIXTURE"]
            for fixture in [args[2]] + (extraFixture.map { [$0] } ?? []) {
                let input = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: fixture))) as! [String: Any]
                let tokens = input["tokens"] as! [Int]
                let expected = input["expected_tokens"] as? [Int]
                let diagnoseLogits = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_DIAGNOSE_LOGITS"] == "1"
                let chunk = Int(ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PREFILL"] ?? "") ?? (input["prefill_step"] as? Int) ?? (tokens.count >= 1024 ? 1024 : tokens.count >= 128 ? 512 : 16)
                print("CASE", fixture, "prompt tokens", tokens.count)
                let comparePrefill = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_COMPARE_PREFILL"] == "1"
                var runs: [(mode: Int, chunk: Int)] = comparePrefill
                    ? [(0, 1), (0, 512), (0, 512)] + (draft == nil ? [] : [(1, 512), (2, 512)])
                    : (draft == nil ? [0] : [0, 1, 2]).map { ($0, chunk) }
                let repetitions = max(1, Int(ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_REPETITIONS"] ?? "1") ?? 1)
                for _ in 1..<repetitions { runs += (draft == nil ? [0] : [0, 1]).map { ($0, chunk) } }
                let modeOrderKey = tokens.count < 128 ? "SYNOMANCER_GGUF_MODE_ORDER" : "SYNOMANCER_GGUF_LONG_MODE_ORDER"
                if let order = ProcessInfo.processInfo.environment[modeOrderKey] {
                    let modes = order.split(separator: ",").map { Int($0)! }
                    precondition(!modes.isEmpty && modes.allSatisfy { (0...2).contains($0) && ($0 == 0 || draft != nil) })
                    runs = modes.map { ($0, chunk) }
                }
                let variants = tokens.count < 128 ? ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_EXPERT_VARIANTS"]?.split(separator: ",").map(String.init) : nil
                if let variants {
                    precondition(variants.allSatisfy { ["plain", "compiled", "gdn", "chain", "early"].contains($0) })
                    runs = variants.flatMap { _ in (draft == nil ? [0] : [0, 1]).map { ($0, chunk) } }
                }
                if tokens.count >= 128, let choices = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PREFILL_CHUNKS"] {
                    let chunks = choices.split(separator: ",").map { Int($0)! }
                    precondition(chunks.allSatisfy { (1...2048).contains($0) })
                    runs = chunks.flatMap { size in (draft == nil ? [0] : [0, 1]).map { ($0, size) } }
                }
                if ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_TRACE"] == "1" {
                    precondition(variants == nil, "Run layer tracing separately from graph variant benchmarks")
                    runs.insert((-1, chunk), at: 0)
                }
                var ordinaryTokens: [Int: [Int]] = [:]
                let strides = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_SUBMISSION_STRIDES"]?.split(separator: ",").map { Int($0)! }
                if let strides {
                    precondition(strides.allSatisfy { (1...48).contains($0) })
                    runs = strides.map { _ in (0, chunk) }
                }
                for (runIndex, run) in runs.enumerated() {
                    model.submissionStride = strides?[runIndex] ?? 2
                    print("SUBMISSION stride", model.submissionStride)
                    let (mode, chunk) = run
                    let variant = variants?[runIndex / (draft == nil ? 1 : 2)]
                    let compiled = variant.map { $0 != "plain" }
                        ?? (ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_COMPILED_EXPERTS"].map { $0 == "1" } ?? true)
                    let gdn = variant.map { $0 == "gdn" || $0 == "chain" || $0 == "early" }
                        ?? (ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_COMPILED_GDN"].map { $0 == "1" } ?? true)
                    let chain = variant.map { $0 == "chain" || $0 == "early" } ?? (ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_DRAFT_CHAIN"].map { $0 == "1" } ?? true)
                    draft?.warmupRounds = variant.map { $0 == "early" ? 2 : 8 }
                        ?? (Int(ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_MTP_WARMUP"] ?? "") ?? (tokens.count >= 128 ? 8 : 2))
                    draft?.usePrefixHitRate = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PREFIX_POLICY"].map { $0 == "1" } ?? true
                    draft?.usePrefixCommit = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PREFIX_COMMIT"].map { $0 == "1" } ?? true
                    draft?.checkPrefixCommit = mode == 2 && ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_CHECK_PREFIX_COMMIT"] == "1"
                    draft?.gpuDraftChain = chain
                    model.compiledExperts = compiled; draft?.draft.compiledExperts = compiled
                    model.compiledGDN = gdn; draft?.draft.compiledGDN = gdn
                    print("CONFIG compiled experts", compiled, "compiled GDN", gdn, "GPU draft chain", chain, "warmup rounds", draft?.warmupRounds ?? 0)
                    model.reset(); draft?.reset()
                    draft?.forceFirstRejection = mode == 2
                    let prefillStart = Date()
                    var logits = MLXArray(0)
                    let separateLast = input["separate_last_token"] as? Bool ?? false
                    let ranges = Qwen4GGUFModel.prefillRanges(count: tokens.count, step: chunk, separateLast: separateLast)
                    let draftLastOnly = input["draft_last_only"] as? Bool ?? false
                    for range in ranges {
                        let start = range.lowerBound, end = range.upperBound
                        logits = model.forward(Array(tokens[start..<end]), prefill: chunk >= 128, lastLogitsOnly: true).reshaped([-1])
                        if mode > 0 && (!draftLastOnly || end == tokens.count) {
                            var shifted = Array(tokens[(start+1)..<min(end+1, tokens.count)])
                            if end == tokens.count { shifted.append(argMax(logits).item(Int.self)) }
                            if draftLastOnly {
                                draft!.prefill(shiftedTokens: [shifted.last!],
                                    hidden: model.lastHidden[0..., (model.lastHidden.dim(1)-1)..., 0...], fast: false)
                            } else {
                                draft!.prefill(shiftedTokens: shifted, hidden: model.lastHidden, fast: chunk >= 128)
                            }
                        }
                        print("prefill", mode, end, "top", argMax(logits).item(Int.self))
                    }
                    print("PREFILL seconds", Date().timeIntervalSince(prefillStart), "chunk", chunk, "mode", mode, "active bytes", MLX.Memory.activeMemory, "cache bytes", MLX.Memory.cacheMemory)
                    if mode == -1 {
                        let expected = expected!
                        for token in expected.prefix(6) { _ = model.step(token) }
                        let snapshot = model.snapshot()
                        var batch: [String: MLXArray] = [:]
                        model.trace = { name, value in eval(value); batch[name] = value }
                        let batched = model.forward(Array(expected[6..<8]))
                        model.restore(snapshot)
                        for row in 0..<2 {
                            model.trace = { name, value in
                                let reference = batch[name]![0..., row..<(row+1), 0...]
                                let delta = max(abs(reference.asType(.float32) - value.asType(.float32))).item(Float.self)
                                precondition(delta == 0, "Verifier hidden mismatch row \(row), \(name), delta \(delta)")
                            }
                            let single = model.step(expected[6+row])
                            print("TRACE logits", row, argMax(single).item(Int.self), argMax(batched[0, row]).item(Int.self))
                        }
                        model.trace = nil
                        continue
                    }
                    var generated: [Int] = []
                    let generationStart = Date()
                    let limit = expected?.count ?? 16
                    var pending = argMax(logits).item(Int.self)
                    while generated.count < limit {
                        if mode == 0 {
                            generated.append(pending)
                            if generated.count == limit { break }
                            let nextLogits = model.step(pending)
                            if diagnoseLogits && tokens.count > 2048 && generated.count == 35 {
                                print("DIAGNOSTIC logits at token 35: Color", nextLogits[1579].item(Float.self),
                                      "Record", nextLogits[6269].item(Float.self))
                            }
                            pending = argMax(nextLogits).item(Int.self)
                        } else {
                            let oldHistoryCount = model.history.count
                            let remaining = limit - generated.count
                            let verified = try draft!.verify(root: pending, target: model, maxCount: limit - generated.count)
                            if remaining == 1 {
                                precondition(verified.tokens == [pending] && verified.next == nil && model.history.count == oldHistoryCount,
                                             "Final pending token must not compute an unused next token")
                            }
                            generated += verified.tokens
                            if let next = verified.next { pending = next }
                        }
                    }
                    let decodeSeconds = Date().timeIntervalSince(generationStart)
                    if let checks = draft?.prefixCommitsChecked, checks > 0 { print("PASS: exact prefix cache/hidden comparisons", checks) }
                    print("tokens:", generated)
                    if let expected {
                        if generated != expected { print("FIRST MISMATCH", zip(generated, expected).enumerated().first(where: { $0.element.0 != $0.element.1 })!) }
                        if ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_DIAGNOSE_PARITY"] != "1" {
                            precondition(generated == expected, "Native tokens differ from reference")
                        }
                        if generated == expected { print("PASS: all \(expected.count) reference tokens match, mode", mode) }
                        if let ordinary = ordinaryTokens[chunk] {
                            precondition(generated == ordinary, "MTP/repeat differs from ordinary native decode")
                            print("PASS: native MTP/repetition consistency, mode", mode)
                        } else if mode == 0 { ordinaryTokens[chunk] = generated }
                    }
                    // The first token comes from prefill, matching the reference benchmark.
                    print("Native decode tokens/s:", Double(max(0, generated.count - 1)) / decodeSeconds, "mode", mode,
                          "proposed", draft?.proposed ?? 0, "accepted", draft?.accepted ?? 0, "rejected rounds", draft?.rejectedRounds ?? 0)
                    if mode == 2 { precondition(draft!.forcedRejections == 1, "Forced rejection not exercised") }
                }
            }
        }
    }
    static func integration(directory: String, image: String) async throws {
        let cache = SpeechCacheAccess.Configuration(path: directory, bookmark: nil)
        let service = LocalChatService()
        let models = try await service.models(cache: cache)
        guard let model = models.first(where: { $0.modelType == "qwen4_exp_gguf" }) else { fatalError("GGUF model not discovered") }
        precondition(model.supportsImages)
        print("PASS: native discovery", model.id)
        let messages: [LocalChatMessage] = [.init(role: "user", content: "The passphrase is REDWOOD-472."), .init(role: "assistant", content: "Understood."), .init(role: "user", content: "Reply only with the passphrase.")]
        let answer = try await service.generate(modelID: model.id, messages: messages, cache: cache, temperature: 0, maxTokens: 32) { _, _ in }
        print("TEXT", answer.text, "tokens/s", answer.tokensPerSecond)
        precondition(answer.text.contains("REDWOOD-472"))
        precondition(answer.mtpFallbackReason == nil && answer.proposedDraftTokens != nil)
        let visual = try await service.generate(modelID: model.id, messages: [.init(role: "user", content: "请简短描述图片中杯子的颜色和把手方向。", image: Data(contentsOf: URL(fileURLWithPath: image)))], cache: cache, temperature: 0, maxTokens: 64) { text, _ in print("VISION", text) }
        print("VISION FINAL", visual.text, "tokens/s", visual.tokensPerSecond, "proposed", visual.proposedDraftTokens ?? 0, "accepted", visual.acceptedDraftTokens ?? 0)
        precondition(visual.text.contains("蓝") && visual.text.contains("右"))
        let plainVisual = try await service.generate(modelID: model.id, messages: [.init(role: "user", content: "请简短描述图片中杯子的颜色和把手方向。", image: Data(contentsOf: URL(fileURLWithPath: image)))], cache: cache, temperature: 0, maxTokens: 64, useMTP: false) { _, _ in }
        precondition(visual.text == plainVisual.text && visual.completionTokens == plainVisual.completionTokens)
        precondition(visual.mtpFallbackReason == nil && visual.proposedDraftTokens != nil)
        print("PASS: vision MTP on/off output and token count match")
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        let cancelled = Task {
            try await service.generate(modelID: model.id, messages: [.init(role: "user", content: "Write a long story about the sea.")], cache: cache, temperature: 0, maxTokens: 1024) { _, _ in continuation.yield(()) }
        }
        for await _ in stream { break }
        cancelled.cancel()
        do { _ = try await cancelled.value; fatalError("Cancellation returned success") }
        catch is CancellationError { print("PASS: native cancellation") }
        continuation.finish()
        let restart = try await service.generate(modelID: model.id, messages: [.init(role: "user", content: "Reply only with OK.")], cache: cache, temperature: 0, maxTokens: 16, stop: ["OK"], useMTP: false) { _, _ in }
        precondition(!restart.text.contains("OK") && restart.finishReason == "stop")
        let sampled = try await service.generate(modelID: model.id, messages: [.init(role: "user", content: "Say hello.")], cache: cache, temperature: 0.7, maxTokens: 4, topP: 0.8, useMTP: true) { _, _ in }
        precondition(sampled.completionTokens > 0 && sampled.mtpFallbackReason == "Qwen MTP requires temperature 0")
        let limited = try await service.generate(modelID: model.id, messages: [.init(role: "user", content: "Count from one to ten.")], cache: cache, temperature: 0, maxTokens: 1) { _, _ in }
        precondition(limited.completionTokens == 1)
        print("PASS: restart, stop sequences, sampling fallback, and one-token MTP limit", sampled.text)
    }

}
