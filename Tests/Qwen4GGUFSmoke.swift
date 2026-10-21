import Foundation
import Darwin
import MLX

@main
struct Qwen4GGUFSmoke {
    static func memoryStatistics(_ model: Qwen4GGUFModel) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        print("MEMORY MLX active", MLX.Memory.activeMemory, "MLX peak", MLX.Memory.peakMemory,
              "expert bytes", model.expertCache?.bytes ?? 0, "expert hits", model.expertCache?.hits ?? 0,
              "expert misses", model.expertCache?.misses ?? 0, "PLE bytes", model.pleCache?.bytes ?? 0,
              "footprint", status == KERN_SUCCESS ? info.phys_footprint : 0)
        if let cache = model.expertCache { precondition(cache.bytes <= cache.budget) }
        if let cache = model.pleCache { precondition(cache.bytes <= cache.budget) }
        precondition(model.temporaryExpandedBytes == 0)
    }
    static func main() async throws {
        setbuf(stdout, nil)
        let args = CommandLine.arguments
        if args.count == 3, args[1] == "--response-probe" {
            let directory = URL(fileURLWithPath: args[2])
            let service = LocalQwen4GGUF(directory: directory)
            for (temperature, mtp) in [(Float(0), false), (Float(0), true), (Float(0.6), false)] {
                let result = try await service.generate(messages: [.init(role: "user", content: "介绍一下你自己")],
                    temperature: temperature, maxTokens: 128, topP: 1, stop: [], useMTP: mtp, prefillStepSize: nil) { _, _ in }
                print("PROBE temperature", temperature, "mtp", mtp, "tokens", result.completionTokens,
                      "finish", result.finishReason, "text", String(reflecting: result.text))
                precondition(!AssistantMessageContent(result.text).answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                             "Self-introduction must produce a visible answer")
            }
            return
        }
        if args.count == 2, args[1] == "--kernels" { try GGUFKernelTests.run(); return }
        if args.count == 4, args[1] == "--integration" { try await integration(directory: args[2], image: args[3]); return }
        guard args.count == 3 else { fatalError("usage: Qwen4GGUFSmoke first-shard.gguf input.json") }
        MLX.Memory.cacheLimit = 1 << 30
        let ticket = WiredMemoryTicket(size: 105 * 1024 * 1024 * 1024, policy: WiredMaxPolicy())
        try await ticket.withWiredLimit {
            let start = Date()
            let resident = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_RESIDENT"] == "1"
            let expertGiB = Int(ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_EXPERT_GIB"] ?? "16")!
            let model = try Qwen4GGUFModel(firstShard: URL(fileURLWithPath: args[1]),
                pagedPLE: !resident, expertBudget: resident || expertGiB < 0 ? nil : expertGiB * 1024 * 1024 * 1024)
            print("MEMORY loaded MLX", MLX.Memory.activeMemory, "expert budget", model.expertCache?.budget ?? -1)
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
                let orderKey = tokens.count < 128 ? "SYNOMANCER_GGUF_PR9_ORDER" : "SYNOMANCER_GGUF_PR9_LONG_ORDER"
                let decodeOrder = (ProcessInfo.processInfo.environment[orderKey] ?? ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_ORDER"])?.split(separator: ",").map(String.init)
                if let decodeOrder {
                    precondition(decodeOrder.allSatisfy { ["b0", "b1", "b2", "o0", "o1", "o2"].contains($0) })
                    runs = decodeOrder.map { (Int(String($0.last!))!, chunk) }
                }
                let maskKey = tokens.count < 128 ? "SYNOMANCER_GGUF_PR9_MASK_ORDER" : "SYNOMANCER_GGUF_PR9_LONG_MASK_ORDER"
                let maskOrder = (ProcessInfo.processInfo.environment[maskKey] ?? ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_MASK_ORDER"])?.split(separator: ",").map { Int($0)! }
                if let maskOrder { precondition(maskOrder.count == runs.count && maskOrder.allSatisfy { (0...15).contains($0) }) }
                if ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_ABLATION_ONLY"] == "1" { runs = [] }
                for (runIndex, run) in runs.enumerated() {
                    if let decodeOrder {
                        Qwen4GGUFDecodeKernels.enabled = decodeOrder[runIndex].first == "o"
                        print("PR9", Qwen4GGUFDecodeKernels.enabled ? "optimized" : "baseline", "run", runIndex)
                    }
                    if let maskOrder { Qwen4GGUFDecodeKernels.mask = maskOrder[runIndex] }
                    print("PR9 MASK", Qwen4GGUFDecodeKernels.mask, "thermal", ProcessInfo.processInfo.thermalState.rawValue)
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
                    eval(logits)
                    let expandedBefore = model.temporaryExpandedBytes
                    model.clearPrefillCaches(); draft?.draft.clearPrefillCaches()
                    MLX.Memory.clearCache()
                    print("MEMORY released expanded", expandedBefore, "expert cache", model.expertCache?.bytes ?? 0)
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
                    memoryStatistics(model)
                    if mode == 2 { precondition(draft!.forcedRejections == 1, "Forced rejection not exercised") }
                    if let pause = Int(ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_BENCH_PAUSE_MS"] ?? ""), pause > 0 {
                        precondition(pause <= 60_000)
                        print("BENCH pause_ms", pause, "thermal", ProcessInfo.processInfo.thermalState.rawValue)
                        try await Task.sleep(nanoseconds: UInt64(pause) * 1_000_000)
                    }
                }
                if ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_ABLATION"] == "1",
                   (tokens.count < 128 || ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_ABLATION_LONG"] != "0"), let expected {
                    ablateDecode(model, prompt: tokens, continuation: expected)
                }
                if ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_PAIRED"] == "1", let expected {
                    pairedDecode(model, prompt: tokens, continuation: expected)
                }
            }
        }
    }
    /// Same state/input, alternate which variant runs first. Timing includes host
    /// dispatch and GPU completion; comparisons happen outside the timed region.
    static func pairedDecode(_ model: Qwen4GGUFModel, prompt: [Int], continuation: [Int]) {
        for length in [1, 4] {
            model.reset()
            let chunk = prompt.count >= 1024 ? 1024 : 16
            for range in Qwen4GGUFModel.prefillRanges(count: prompt.count, step: chunk, separateLast: false) {
                eval(model.forward(Array(prompt[range]), prefill: chunk >= 128))
            }
            model.clearPrefillCaches()
            MLX.Memory.clearCache()
            for sample in 0..<min(28, continuation.count / length) {
                let snapshot = model.snapshot()
                let input = Array(continuation[(sample*length)..<((sample+1)*length)])
                var outputs: [Bool: MLXArray] = [:]
                var times: [Bool: Double] = [:]
                for optimized in (sample % 2 == 0 ? [false, true] : [true, false]) {
                    model.restore(snapshot)
                    Qwen4GGUFDecodeKernels.enabled = optimized
                    let start = Date()
                    let output = model.forward(input)
                    eval(output)
                    times[optimized] = Date().timeIntervalSince(start) * 1000
                    outputs[optimized] = output
                }
                let delta = max(abs(outputs[false]! - outputs[true]!)).item(Float.self)
                print("PAIRED tokens", length, "sample", sample, "baseline_ms", times[false]!,
                      "optimized_ms", times[true]!, "max_delta", delta)
                precondition(delta == 0, "PR9 changed fixed-input logits")
            }
        }
    }

    /// Balanced cyclic/reversed orders avoid attributing temporal drift to a feature.
    /// Every candidate sees the identical KV/recurrent state and token window.
    static func ablateDecode(_ model: Qwen4GGUFModel, prompt: [Int], continuation: [Int]) {
        let originalMask = Qwen4GGUFDecodeKernels.mask
        defer { Qwen4GGUFDecodeKernels.mask = originalMask }
        let masks = ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_ABLATION_MASKS"]?
            .split(separator: ",").map { Int($0)! } ?? [0, 1, 2, 4, 8, 15, 14, 13, 11, 7]
        precondition(masks.contains(0) && masks.allSatisfy { (0...15).contains($0) })
        for length in [1, 2, 4] {
            model.reset()
            Qwen4GGUFDecodeKernels.mask = 0
            let chunk = prompt.count >= 1024 ? 1024 : 16
            for range in Qwen4GGUFModel.prefillRanges(count: prompt.count, step: chunk, separateLast: false) {
                eval(model.forward(Array(prompt[range]), prefill: chunk >= 128))
            }
            model.clearPrefillCaches(); MLX.Memory.clearCache()
            let samples = Int(ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_ABLATION_SAMPLES"] ?? "64")!
            precondition(samples > 4)
            for sample in 0..<min(samples, continuation.count / length) {
                let snapshot = model.snapshot()
                let input = Array(continuation[(sample*length)..<((sample+1)*length)])
                let shift = sample % masks.count
                var order = Array(masks[shift...]) + Array(masks[..<shift])
                if (sample / masks.count) % 2 != 0 { order.reverse() }
                var outputs: [Int: MLXArray] = [:]
                var times: [Int: Double] = [:]
                for mask in order {
                    model.restore(snapshot)
                    Qwen4GGUFDecodeKernels.mask = mask
                    let start = Date()
                    let output = model.forward(input)
                    eval(output)
                    times[mask] = Date().timeIntervalSince(start) * 1000
                    outputs[mask] = output
                }
                for mask in masks {
                    let delta = max(abs(outputs[0]! - outputs[mask]!)).item(Float.self)
                    print("ABLATION window", length, "sample", sample, "mask", mask,
                          "ms", times[mask]!, "delta_ms", times[mask]! - times[0]!,
                          "logit_delta", delta, "thermal", ProcessInfo.processInfo.thermalState.rawValue)
                    precondition(delta == 0, "Ablation changed fixed-input logits")
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
        let memoryBeforeUnload = MLX.Memory.activeMemory
        enum ImageProbeError: Error { case expected }
        do {
            try await service.withUnloadedChat {
                precondition(MLX.Memory.activeMemory + 1024 * 1024 * 1024 < memoryBeforeUnload,
                             "Image reservation must release loaded chat weights")
                let (waiting, signal) = AsyncStream<Void>.makeStream()
                let queuedChat = Task {
                    try await service.generate(modelID: "nonexistent-model", messages: messages, cache: cache,
                        onProgress: { _, _ in signal.yield(()) }) { _, _ in }
                }
                // The chat has entered generate(), but must not reach model
                // validation while this image reservation is held.
                for await _ in waiting { break }
                queuedChat.cancel()
                do { _ = try await queuedChat.value; fatalError("Queued chat unexpectedly completed") }
                catch is CancellationError { }
                signal.finish()
                throw ImageProbeError.expected
            }
        } catch ImageProbeError.expected { }
        print("PASS: image reservation unloads chat, blocks new chat and releases its gate after error")
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
