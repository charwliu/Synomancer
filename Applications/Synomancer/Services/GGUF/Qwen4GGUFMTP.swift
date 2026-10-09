import Foundation
import MLX

/// Greedy native MTP. Only target-verified tokens are emitted.
/// Rejection trims draft KV and commits target prefix state without repeating model projections.
final class Qwen4GGUFMTP {
    let draft: Qwen4GGUFModel
    private var seed: Int = 0
    private var seedHidden = MLXArray(0)
    private(set) var proposed = 0
    private(set) var accepted = 0
    private(set) var rejectedRounds = 0
    private(set) var forcedRejections = 0
    private var round = 0
    // Used by regression tests to exercise the otherwise prompt-dependent rejection path.
    var forceFirstRejection = false
    var gpuDraftChain = true
    var warmupRounds = 2
    var usePrefixHitRate = true
    var usePrefixCommit = true
    var checkPrefixCommit = false
    private var recentAcceptances: [Int] = []
    private(set) var prefixCommitsChecked = 0

    init(sidecar: URL) throws { draft = try Qwen4GGUFModel(mtpSidecar: sidecar) }
    func reset() {
        draft.reset(); proposed = 0; accepted = 0; rejectedRounds = 0; forcedRejections = 0; round = 0
        seedHidden = MLXArray(0)
        recentAcceptances = []
        prefixCommitsChecked = 0
    }
    func prefill(shiftedTokens: [Int], hidden: MLXArray, fast: Bool) {
        seed = argMax(draft.draftForward(shiftedTokens, hidden: hidden, prefill: fast)).item(Int.self)
        seedHidden = draft.lastHidden
    }

    /// Expand only when the trained one-token prefix is useful often enough;
    /// failed tail tokens must not be counted as failed prefix predictions.
    static func adaptiveDraftCount(round: Int, warmupRounds: Int, acceptances: [Int]) -> Int {
        let recent = acceptances.suffix(32)
        guard round >= warmupRounds, !recent.isEmpty else { return 1 }
        let hits = recent.filter { $0 > 0 }.count
        return Double(hits) / Double(recent.count) >= 0.65 ? 3 : 1
    }

    /// The root is the pending, already-target-sampled token. Returns root + accepted drafts
    /// and the next pending target token. nil marks the last permitted token;
    /// no unused next-token forward pass is needed at that request boundary.
    func verify(root: Int, target: Qwen4GGUFModel, maxCount: Int) throws -> (tokens: [Int], next: Int?) {
        try Task.checkCancellation()
        precondition(maxCount > 0)
        if maxCount == 1 { return ([root], nil) }
        let acceptance = proposed == 0 ? 0 : Double(accepted) / Double(proposed)
        let budget = usePrefixHitRate
            ? Self.adaptiveDraftCount(round: round, warmupRounds: warmupRounds, acceptances: recentAcceptances)
            : (round < warmupRounds || acceptance < 0.6 ? 1 : 3)
        let limit = min(max(0, maxCount - 1), budget)
        let initialDraftCount = draft.draftPosition
        var candidates = [seed]
        var hidden = seedHidden
        if gpuDraftChain && limit > 1 {
            var pending = MLXArray(seed)
            var predictions: [MLXArray] = []
            for _ in 1..<limit {
                let logits = draft.draftForward(pending, count: 1, hidden: hidden)
                hidden = draft.lastHidden
                pending = argMax(logits)
                predictions.append(pending)
            }
            candidates += stacked(predictions).asArray(Int.self)
        } else {
            for _ in 1..<limit {
                let logits = draft.draftForward([candidates.last!], hidden: hidden)
                hidden = draft.lastHidden
                candidates.append(argMax(logits).item(Int.self))
            }
        }
        let snapshot = target.snapshot()
        let logits = target.forward([root] + candidates, capturePrefix: usePrefixCommit)
        let decisions = argMax(logits, axis: -1).asArray(Int.self)
        if forceFirstRejection && forcedRejections == 0 && candidates[0] == decisions[0] {
            candidates[0] = (decisions[0] + 1) % 248320
            forcedRejections += 1
        }
        let verifiedHidden = target.lastHidden
        var count = 0
        while count < candidates.count && candidates[count] == decisions[count] { count += 1 }
        let next = decisions[count]
        let committed = [root] + Array(candidates.prefix(count))
        if count < candidates.count { rejectedRounds += 1 }
        if usePrefixCommit {
            target.commitVerifiedPrefix(from: snapshot, count: committed.count)
            if checkPrefixCommit && count < candidates.count {
                let direct = target.snapshot()
                target.restore(snapshot)
                _ = target.forward(committed)
                precondition(direct.history == target.history && all(direct.hidden .== target.lastHidden).item(Bool.self),
                             "Prefix commit changed target hidden/history")
                for (actual, expected) in zip(direct.caches, target.caches) {
                    let a = [actual.conv, actual.recurrent, actual.keys, actual.values, actual.indexKeys, actual.indexBlocks, actual.pleConv]
                    let b = [expected.conv, expected.recurrent, expected.keys, expected.values, expected.indexKeys, expected.indexBlocks, expected.pleConv]
                    for (x, y) in zip(a, b) {
                        if let x, let y { precondition(x.shape == y.shape && all(x .== y).item(Bool.self), "Prefix commit changed cache") }
                        else { precondition(x == nil && y == nil, "Prefix commit changed cache presence") }
                    }
                }
                prefixCommitsChecked += 1
                target.restore(direct)
            }
        } else if count < candidates.count {
            target.restore(snapshot)
            _ = target.forward(committed)
        }
        // Keep only speculative draft cache entries whose tokens were accepted.
        let keep = min(count, limit - 1)
        draft.trimDraft(to: initialDraftCount + keep)
        var updateTokens = Array(candidates[keep..<count])
        updateTokens.append(next)
        let hiddens = verifiedHidden[0..., keep..<(count+1), 0...]
        seed = argMax(draft.draftForward(updateTokens, hidden: hiddens)).item(Int.self)
        seedHidden = draft.lastHidden
        proposed += candidates.count; accepted += count; round += 1
        recentAcceptances.append(count)
        if recentAcceptances.count > 32 { recentAcceptances.removeFirst() }
        try Task.checkCancellation()
        return (committed, next)
    }
}
