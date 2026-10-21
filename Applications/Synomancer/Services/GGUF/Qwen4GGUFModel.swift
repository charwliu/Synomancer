/*
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Qwen4-exp single-sequence inference using original mixed GGUF quantization.
/// Adapted from mlx-vlm c24454765c3177a385185c185b4957567c5fdf89 (MIT).
final class Qwen4GGUFModel {
    struct Cache {
        var conv: MLXArray?
        var recurrent: MLXArray?
        var keys: MLXArray?
        var values: MLXArray?
        var indexKeys: MLXArray?
        var indexBlocks: MLXArray?
        var pleConv: MLXArray?
    }
    let sparseAttention = Qwen4GGUFSparseAttention()
    let files: [GGUFFile]
    private(set) var expertCache: GGUFWeightCache?
    private(set) var pleCache: GGUFWeightCache?
    var temporaryExpandedBytes: Int { weights.values.reduce(0) { $0 + $1.temporaryExpandedBytes } }
    func clearPrefillCaches() {
        // Call after evaluated prefill outputs, or at the next decode boundary.
        for weight in weights.values { weight.clearPrefillCache() }
    }
    var weights: [String: GGUFTensor] = [:]
    var denseWeights: [String: MLXArray] = [:]
    var caches = (0..<48).map { _ in Cache() }
    var lastHidden = MLXArray(0)
    var trace: ((String, MLXArray) -> Void)?
    var fastPrefill = false
    private var capturingPrefix = false
    private var gdnReplay: [Int: [MLXArray]] = [:]
    private var pleReplay: [MLXArray] = []
    // Coalesce singleton decode submission to reduce CPU/Metal scheduling overhead.
    // Prefill and multi-row verification keep their per-layer submission.
    var submissionStride = 2
    var compiledExperts = true
    var compiledGDN = true
    private var expertGraphs: [String: @Sendable (MLXArray) -> MLXArray] = [:]
    private var gdnGraphs: [Int: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
    private var gdnParameters: [Int: GDNGraphWeights] = [:]
    private var expertWeights: [String: [GGUFTensor]] = [:]
    // The model is confined to the service's serial generation gate. MLX traces
    // this callback synchronously; the captured tensor caches are never shared
    // with another model/request executing concurrently.
    private struct ExpertGraphWeights: @unchecked Sendable { let values: [GGUFTensor] }
    struct GDNGraphWeights: @unchecked Sendable {
        let qkv: GGUFTensor
        let gate: GGUFTensor
        let alpha: GGUFTensor
        let beta: GGUFTensor
        let projection: GGUFTensor
        let convolution: MLXArray
        let aLog: MLXArray
        let bias: MLXArray
        let norm: MLXArray
    }
    var history = [Int]()
    private(set) var draftPosition = 0
    var positionIDs: [[Int]] = []
    var imageEmbeddings: [Int: MLXArray] = [:]
    let multipliers: [Int64]
    let offsets: [Int64]
    let vocabSizes: [Int64]

    init(firstShard: URL, pagedPLE: Bool = true, expertBudget: Int? = GGUFMemoryPolicy.expertBudget(), onProgress: @Sendable (Double) -> Void = { _ in }) throws {
        let second = firstShard.deletingLastPathComponent().appendingPathComponent(firstShard.lastPathComponent.replacingOccurrences(of: "00001-of", with: "00002-of"))
        guard firstShard != second else { throw GGUFFile.Failure(message: "Expected first split IQ3_S shard") }
        files = try [GGUFFile(url: firstShard), GGUFFile(url: second)]
        let metadata = files[0].metadata
        guard metadata["general.architecture"] as? String == "qwen4exp",
              (metadata["qwen4exp.block_count"] as? UInt32) == 48 else {
            throw GGUFFile.Failure(message: "Expected 48-layer Qwen4-exp checkpoint")
        }
        func integers(_ key: String) throws -> [Int64] {
            guard let values = metadata["qwen4exp.ple." + key] as? [Any] else { throw GGUFFile.Failure(message: "Missing PLE \(key)") }
            return try values.map {
                if let n = $0 as? Int64 { return n }
                if let n = $0 as? UInt64 { return Int64(bitPattern: n) }
                if let n = $0 as? UInt32 { return Int64(n) }
                throw GGUFFile.Failure(message: "Invalid PLE metadata")
            }
        }
        multipliers = try integers("layer_multipliers")
        offsets = try integers("head_offsets")
        vocabSizes = try integers("head_vocab_sizes")
        guard multipliers.count == 3, offsets.count == 16, vocabSizes.count == 16,
              zip(offsets, vocabSizes).allSatisfy({ $0 >= 0 && $1 > 0 && $0 < 320001536 && $1 <= 320001536 - $0 }) else {
            throw GGUFFile.Failure(message: "Unsupported PLE shape")
        }
        try Self.validate(files)
        expertCache = expertBudget.map { GGUFWeightCache(budget: $0) }
        pleCache = pagedPLE ? GGUFWeightCache(budget: 16 * 1024 * 1024, maximumEntries: 8192) : nil
        func cache(for tensor: GGUFFile.Tensor) -> GGUFWeightCache? {
            if tensor.name == "per_layer_token_embd.weight" { return pleCache }
            if tensor.name.hasSuffix("_exps.weight") { return expertCache }
            return nil
        }
        let totalBytes = files.reduce(0) { $0 + $1.tensors.values.reduce(0) { $0 + (cache(for: $1) == nil ? $1.byteCount : 0) } }
        var loadedBytes = 0
        var lastPercent = -1
        onProgress(0)
        for file in files {
            for tensor in file.tensors.values {
                try Task.checkCancellation()
                guard weights[tensor.name] == nil else { throw GGUFFile.Failure(message: "Duplicate shard tensor") }
                weights[tensor.name] = try GGUFTensor(file: file, tensor: tensor, cache: cache(for: tensor))
                loadedBytes += cache(for: tensor) == nil ? tensor.byteCount : 0
                let percent = Int(100 * Double(loadedBytes) / Double(max(1, totalBytes)))
                if percent != lastPercent {
                    lastPercent = percent
                    onProgress(Double(loadedBytes) / Double(max(1, totalBytes)))
                }
            }
        }
    }

    init(mtpSidecar: URL) throws {
        let file = try GGUFFile(url: mtpSidecar)
        try Self.validateMTP(file)
        files = [file]
        multipliers = []; offsets = []; vocabSizes = []
        caches = (0..<49).map { _ in Cache() }
        for tensor in file.tensors.values where !tensor.name.hasPrefix("output_hc_") {
            try Task.checkCancellation()
            weights[tensor.name] = try GGUFTensor(file: file, tensor: tensor)
        }
        for part in ["norm", "down", "up"] {
            weights["output_hc_" + part + ".weight"] = weights["blk.48.nextn.hc_head_" + part + ".weight"]
        }
    }

    static func prefillRanges(count: Int, step: Int, separateLast: Bool) -> [Range<Int>] {
        precondition(count > 0 && step > 0)
        let prefixCount = separateLast ? count - 1 : count
        var ranges = stride(from: 0, to: prefixCount, by: step).map { $0..<min($0 + step, prefixCount) }
        if separateLast { ranges.append(prefixCount..<count) }
        return ranges
    }

    func reset() {
        gdnReplay.removeAll(); pleReplay.removeAll(); capturingPrefix = false
        caches = caches.map { _ in Cache() }
        lastHidden = MLXArray(0)
        history = []
        draftPosition = 0
        positionIDs = []
        imageEmbeddings = [:]
    }
    func tensor(_ name: String) -> GGUFTensor { weights[name]! }
    func dense(_ name: String) -> MLXArray {
        if let value = denseWeights[name] { return value }
        let value = tensor(name).dense()
        eval(value)
        denseWeights[name] = value
        return value
    }
    private func constant(_ name: String, make: () -> MLXArray) -> MLXArray {
        if let value = denseWeights[name] { return value }
        let value = make()
        eval(value)
        denseWeights[name] = value
        return value
    }
    func linear(_ x: MLXArray, _ name: String) -> MLXArray { tensor(name + ".weight").linear(x, prefill: fastPrefill) }
    func norm(_ x: MLXArray, _ name: String, group: Int? = nil) -> MLXArray {
        let shape = x.shape
        let input = group.map { x.reshaped(Array(x.shape.dropLast()) + [-1, $0]) } ?? x
        let weight = dense(name + ".weight").asType(.float32)
        let w = group.map { weight.reshaped([-1, $0]) } ?? weight
        return Qwen4GGUFOps.norm(input, w).reshaped(shape)
    }
    func hyper(_ x: MLXArray, _ prefix: String) -> (MLXArray, MLXArray) {
        let normalized = norm(x, prefix + "_norm", group: 2560)
        let reduced = Qwen4GGUFOps.activate(linear(normalized, prefix + "_down"))
        let result = Qwen4GGUFOps.mix(linear(reduced, prefix + "_up"), normalized)
        let injection = weights[prefix + "_inject.weight"] == nil ? MLXArray(0) : Qwen4GGUFOps.injection(linear(normalized, prefix + "_inject"))
        return (result, injection)
    }
    func inject(_ x: MLXArray, _ branch: MLXArray, _ gates: MLXArray) -> MLXArray {
        Qwen4GGUFOps.inject([x, branch, gates])[0]
    }
    func moe(_ x: MLXArray, _ p: String) -> MLXArray {
        let graphKey = p + "pr9-\(Qwen4GGUFDecodeKernels.mask & 9)"
        if expertCache == nil && compiledExperts && !fastPrefill, let graph = expertGraphs[graphKey] { return graph(x) }
        let weights: [GGUFTensor]
        if let cached = expertWeights[p] { weights = cached }
        else {
            let names = ["ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
                         "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp_shexp"]
            weights = names.map { tensor(p + $0 + ".weight") }
            expertWeights[p] = weights
        }
        if expertCache == nil && compiledExperts && !fastPrefill {
            // Capture tensors, not the model, so graph ownership cannot retain
            // the model through a cycle. Request caches are not part of MoE.
            let captured = ExpertGraphWeights(values: weights)
            let graph = compile { input in Qwen4GGUFModel.experts(input, weights: captured.values, prefill: false) }
            expertGraphs[graphKey] = graph
            return graph(x)
        }
        return Self.experts(x, weights: weights, prefill: fastPrefill)
    }
    static func experts(_ x: MLXArray, weights: [GGUFTensor], prefill: Bool) -> MLXArray {
        let gates = softmax(weights[0].linear(x, prefill: prefill), axis: -1, precise: true)
        let ids = Qwen4GGUFDecodeKernels.top10(gates)
        var scores = takeAlong(gates, ids, axis: -1)
        scores = scores / sum(scores, axis: -1, keepDims: true)
        let selectedIDs: MLXArray
        let selectedWeights: [GGUFTensor]
        let expertCount: Int
        if weights[1].isFileBacked {
            // One GPU-to-host routing boundary per layer. Sorting preserves
            // expert order for grouped prefill; remapping preserves top-k order.
            let original = ids.asArray(Int.self)
            let unique = Array(Set(original)).sorted()
            let mapping = Dictionary(uniqueKeysWithValues: unique.enumerated().map { ($1, UInt32($0)) })
            selectedIDs = MLXArray(original.map { mapping[$0]! }).reshaped(ids.shape)
            selectedWeights = (1...3).map { weights[$0].selectedRecords(unique, expert: true) }
            expertCount = unique.count
        } else {
            selectedIDs = ids; selectedWeights = Array(weights[1...3]); expertCount = 512
        }
        let routing = prefill && x.dim(1) >= 128 ? GGUFExpertRoutes(ids: selectedIDs, experts: expertCount) : nil
        let gate = selectedWeights[0].gathered(x, ids: selectedIDs, repeats: 10, routing: routing)
        let up = selectedWeights[1].gathered(x, ids: selectedIDs, repeats: 10, routing: routing)
        let output = selectedWeights[2].gathered(Qwen4GGUFOps.swiglu(gate, up), ids: selectedIDs, repeats: 1, routing: routing)
        let shared = weights[6].linear(Qwen4GGUFOps.swiglu(weights[4].linear(x, prefill: prefill), weights[5].linear(x, prefill: prefill)), prefill: prefill)
        let result = Qwen4GGUFDecodeKernels.combine(output, scores: scores, gate: weights[7].linear(x, prefill: prefill), shared: shared)
        // Complete this layer before staging the next one. Otherwise lazy
        // graphs retain every evicted expert until the full forward is read.
        if weights[1].isFileBacked { eval(result) }
        return result
    }
    // GGUF stores GDN value heads repeat-major; native MLX uses key-head-major.
    func inverseHeads(_ x: MLXArray, dim: Int) -> MLXArray {
        let leading = Array(x.shape.dropLast())
        return x.reshaped(leading + [3, 16, dim]).swappedAxes(-3, -2).reshaped(leading + [48 * dim])
    }
    func gdn(_ x: MLXArray, _ i: Int) -> MLXArray {
        let p = "blk.\(i).", length = x.dim(1)
        let parameters: GDNGraphWeights
        if let cached = gdnParameters[i] { parameters = cached }
        else {
            let conv = constant(p + "native_conv_grouped") {
                let original = dense(p + "ssm_conv1d.weight").asType(.float16).transposed()
                return concatenated([original[0..., ..<4096], inverseHeads(original[0..., 4096...], dim: 128)], axis: -1)
                    .transposed().reshaped([10240, 4, 1])
            }
            parameters = GDNGraphWeights(qkv: tensor(p + "attn_qkv.weight"), gate: tensor(p + "attn_gate.weight"),
                alpha: tensor(p + "ssm_alpha.weight"), beta: tensor(p + "ssm_beta.weight"),
                projection: tensor(p + "ssm_out.weight"), convolution: conv,
                aLog: constant(p + "native_a_log") { log(-inverseHeads(dense(p + "ssm_a"), dim: 1)) },
                bias: constant(p + "native_dt_bias") { inverseHeads(dense(p + "ssm_dt.bias"), dim: 1) },
                norm: dense(p + "ssm_norm.weight"))
            gdnParameters[i] = parameters
        }
        let convState = caches[i].conv ?? MLXArray.zeros([1, 3, 10240], dtype: x.dtype)
        let capture = capturingPrefix
        let graphKey = i * 16 + (capture ? 8 : 0) + (Qwen4GGUFDecodeKernels.mask & 6)
        let values: [MLXArray]
        if compiledGDN && !fastPrefill && length <= 8, let recurrent = caches[i].recurrent {
            let graph: @Sendable ([MLXArray]) -> [MLXArray]
            if let cached = gdnGraphs[graphKey] { graph = cached }
            else {
                graph = compile { (inputs: [MLXArray]) in
                    Qwen4GGUFModel.gdnOutput(inputs[0], conv: inputs[1], state: inputs[2], weights: parameters, prefill: false, captureInputs: capture)
                }
                gdnGraphs[graphKey] = graph
            }
            values = graph([x, convState, recurrent])
        } else {
            values = Self.gdnOutput(x, conv: convState, state: caches[i].recurrent, weights: parameters, prefill: fastPrefill, captureInputs: capture)
        }
        if capture { gdnReplay[i] = Array(values.dropFirst(3)) }
        caches[i].conv = values[1]; caches[i].recurrent = values[2]
        return values[0]
    }
    /// State is an explicit input/output, so tracing cannot capture a request's
    /// recurrence or interfere with speculative snapshots and rollback.
    static func gdnOutput(_ x: MLXArray, conv state: MLXArray, state recurrent: MLXArray?,
                          weights: GDNGraphWeights, prefill: Bool, captureInputs: Bool = false) -> [MLXArray] {
        let length = x.dim(1)
        let qkv = weights.qkv.linear(x, prefill: prefill)
        let z = weights.gate.linear(x, prefill: prefill).reshaped([1, length, 48, 128])
        let a = weights.alpha.linear(x, prefill: prefill)
        let b = weights.beta.linear(x, prefill: prefill)
        let input: MLXArray, conv: MLXArray
        if let fused = Qwen4GGUFDecodeKernels.convSilu(qkv, state: state, weight: weights.convolution) {
            (conv, input) = fused
        } else {
            input = concatenated([state, qkv], axis: 1)
            conv = silu(conv1d(input, weights.convolution, groups: 10240))
        }
        let q = conv[.ellipsis, ..<2048].reshaped([1, length, 16, 128])
        let k = conv[.ellipsis, 2048..<4096].reshaped([1, length, 16, 128])
        let v = conv[.ellipsis, 4096...].reshaped([1, length, 48, 128])
        let normalizedQK = Qwen4GGUFOps.normalizeQK([q, k])
        let (output, nextState) = gatedDeltaUpdate(q: normalizedQK[0], k: normalizedQK[1], v: v,
            a: a, b: b, aLog: weights.aLog, dtBias: weights.bias, state: recurrent)
        let gated = Qwen4GGUFDecodeKernels.gatedNorm(output, weight: weights.norm, gate: z).asType(x.dtype)
        let projectedInput = weights.projection.nativeGDNHeads ? gated.reshaped([1, length, 6144])
            : gated.reshaped([1, length, 16, 3, 128]).swappedAxes(-3, -2).reshaped([1, length, 6144])
        let result = [weights.projection.linear(projectedInput, prefill: prefill), input[0..., length..., 0...], nextState]
        return captureInputs ? result + [normalizedQK[0], normalizedQK[1], v, a, b, input] : result
    }
    /// Reuse verifier projections; only the small recurrence needs replay after
    /// rejecting a suffix. No attention, embedding or expert matmul is repeated.
    static func gdnPrefixState(_ inputs: [MLXArray], count: Int, state: MLXArray?,
                               weights: GDNGraphWeights) -> [MLXArray] {
        precondition(inputs.count == 6 && count > 0 && count <= inputs[0].dim(1))
        let (_, nextState) = gatedDeltaUpdate(q: inputs[0][0..., ..<count], k: inputs[1][0..., ..<count],
            v: inputs[2][0..., ..<count], a: inputs[3][0..., ..<count], b: inputs[4][0..., ..<count],
            aLog: weights.aLog, dtBias: weights.bias, state: state)
        return [inputs[5][0..., count..<(count + 3), 0...], nextState]
    }

    func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        guard !positionIDs.isEmpty else {
            return MLXFast.RoPE(x, dimensions: 64, traditional: false, base: 10_000_000, scale: 1, offset: offset)
        }
        let positions = (offset..<(offset+x.dim(-2))).map { index in
            index < positionIDs.count ? positionIDs[index] : Array(repeating: positionIDs.last!.max()! + index - positionIDs.count + 1, count: 3)
        }
        return multimodalRope(x, positions: positions)
    }
    private func multimodalRope(_ x: MLXArray, positions: [[Int]]) -> MLXArray {
        let selected = MLXArray(positions.flatMap { position in (0..<32).map { Float(position[$0 % 3]) } }).reshaped([1, 1, positions.count, 32])
        let inverse = pow(MLXArray(Float(10_000_000)), -MLXArray(0..<32).asType(.float32) / 32)
        let angles = selected * inverse
        let cosine = concatenated([cos(angles), cos(angles)], axis: -1).asType(x.dtype)
        let sine = concatenated([sin(angles), sin(angles)], axis: -1).asType(x.dtype)
        let rotated = concatenated([-x[.ellipsis, 32..<64], x[.ellipsis, ..<32]], axis: -1)
        return concatenated([x[.ellipsis, ..<64] * cosine + rotated * sine, x[.ellipsis, 64...]], axis: -1)
    }

    func attention(_ x: MLXArray, _ i: Int) -> MLXArray {
        let p = "blk.\(i).", position = i == 48 ? draftPosition : history.count, length = x.dim(1)
        let total = position + length
        let raw = linear(x, p + "attn_q").reshaped([1, length, 24, 512])
        let gate = raw[.ellipsis, 256...].reshaped([1, length, 6144])
        let q = rope(norm(raw[.ellipsis, ..<256], p + "attn_q_norm").transposed(0, 2, 1, 3), offset: position)
        var k = rope(norm(linear(x, p + "attn_k").reshaped([1, length, 2, 256]), p + "attn_k_norm").transposed(0, 2, 1, 3), offset: position)
        var v = linear(x, p + "attn_v").reshaped([1, length, 2, 256]).transposed(0, 2, 1, 3)
        if let old = caches[i].keys { k = concatenated([old, k], axis: 2) }
        if let old = caches[i].values { v = concatenated([old, v], axis: 2) }
        caches[i].keys = k; caches[i].values = v
        var indexKeys = linear(x, p + "indexer.k_proj")
        if let old = caches[i].indexKeys { indexKeys = concatenated([old, indexKeys], axis: 1) }
        caches[i].indexKeys = indexKeys
        let queryEnds = MLXArray((position+1)...total).reshaped([1, length, 1])
        let blocks = total / 4
        var selectedBlocks: MLXArray?
        if blocks > 512 {
            let query = rope(norm(linear(x, p + "indexer.q_proj").reshaped([1, length, 4, 128]), p + "indexer.q_norm").transposed(0, 2, 1, 3), offset: position)
            // Completed block keys are immutable. Reuse normalized/rotated history
            // instead of reprocessing the entire prefix on every decode step.
            let oldBlocks = caches[i].indexBlocks?.dim(2) ?? 0
            if blocks > oldBlocks {
                let pooled = mean(indexKeys[0..., (oldBlocks * 4)..<(blocks * 4), 0...].asType(.float32)
                    .reshaped([1, blocks - oldBlocks, 4, 128]), axis: 2).asType(x.dtype)
                let normalized = norm(pooled, p + "indexer.k_norm").reshaped([1, 1, blocks - oldBlocks, 128])
                let appended: MLXArray
                if positionIDs.isEmpty {
                    appended = MLXFast.RoPE(normalized, dimensions: 64, traditional: false,
                        base: 10_000_000, scale: 4, offset: oldBlocks)
                } else {
                    let positions = (oldBlocks..<blocks).map { block -> [Int] in
                        let offset = block * 4
                        return offset < positionIDs.count ? positionIDs[offset] : Array(repeating: positionIDs.last!.max()! + offset - positionIDs.count + 1, count: 3)
                    }
                    appended = multimodalRope(normalized, positions: positions)
                }
                caches[i].indexBlocks = caches[i].indexBlocks.map { concatenated([$0, appended], axis: 2) } ?? appended
            }
            let keys = caches[i].indexBlocks!
            let completeCounts = (queryEnds / 4).asType(.int32)
            let scores = sum(maximum(matmul(query.asType(.float32), keys.asType(.float32).swappedAxes(-1, -2)), 0), axis: 1) / sqrt(Float(128))
            let valid = MLXArray(0..<blocks).reshaped([1, 1, blocks]) .< completeCounts
            let masked = which(valid, scores, -Float.infinity)
            let selected = argPartition(masked, kth: -512, axis: -1)[.ellipsis, (blocks-512)...].asType(.int32)
            selectedBlocks = which(selected .< completeCounts, selected, -1)
        }
        let output: MLXArray
        if length > 8 {
            // Bulk prefill benefits from tiled matrix attention. Indexed vector
            // attention is faster for decode/verifier blocks on this backend.
            let tokenIndices = MLXArray(0..<total).reshaped([1, 1, total])
            let causal = tokenIndices .< queryEnds
            var mask = causal
            if let selectedBlocks {
                let completeCounts = (queryEnds / 4).asType(.int32)
                let indices = (selectedBlocks.expandedDimensions(axis: -1) * 4 + MLXArray(0..<4)).reshaped([1, length, -1])
                let valid = (indices .>= 0) .&& (indices .< total)
                let safeIndices = which(valid, indices, total)
                let scattered = putAlong(MLXArray.zeros([1, length, total+1], dtype: .bool),
                    safeIndices, values: valid, axis: -1)[.ellipsis, ..<total]
                let sparse = (scattered .|| (tokenIndices .>= completeCounts * 4)) .&& causal
                mask = which(completeCounts .> 512, sparse, causal)
            }
            output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 1 / sqrt(Float(256)), mask: mask.reshaped([1, 1, length, total]))
        } else if let selectedBlocks, (length == 1 && total / 4 > 512) || position / 4 > 512 {
            output = sparseAttention(q, k, v, blocks: selectedBlocks, queryEnds: queryEnds.reshaped([1, length]))
        } else if length > 1 && length <= 8 {
            // Verifier rows must use the same SDPA dispatch as singleton decode.
            // A masked multi-query kernel changes FP16 reduction order.
            output = concatenated((0..<length).map { row in
                let end = position + row + 1
                if end / 4 > 512, let selectedBlocks {
                    return sparseAttention(q[0..., 0..., row..<(row+1), 0...],
                        k[0..., 0..., ..<end, 0...], v[0..., 0..., ..<end, 0...],
                        blocks: selectedBlocks[0..., row..<(row+1), 0...], queryEnds: MLXArray([Int32(end)]).reshaped([1, 1]))
                }
                return MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., row..<(row+1), 0...],
                    keys: k[0..., 0..., ..<end, 0...], values: v[0..., 0..., ..<end, 0...],
                    scale: 1 / sqrt(Float(256)), mask: nil)
            }, axis: 2)
        } else {
            output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(256)), mask: nil)
        }
        return linear(Qwen4GGUFOps.sigmoidGate(output.transposed(0, 2, 1, 3).reshaped([1, length, 6144]), gate), p + "attn_output")
    }
    func ple(_ x: MLXArray, tokens: [Int]) -> MLXArray {
        if (2...8).contains(tokens.count) {
            // Preserve singleton FP16 convolution rounding in speculative verification.
            let originalHistory = history
            defer { history = originalHistory }
            var rows: [MLXArray] = []
            for (row, token) in tokens.enumerated() {
                rows.append(ple(x[0..., row..<(row+1), 0...], tokens: [token]))
                history.append(token)
            }
            return concatenated(rows, axis: 1)
        }
        let eos = 248044
        var ids: [UInt32] = []
        var prefix = history
        for token in tokens {
            var previous = [token]
            for t in prefix.reversed().prefix(2) {
                if t == eos { break }
                previous.append(t)
            }
            while previous.count < 3 { previous.append(eos) }
            let products = zip(previous, multipliers).map { Int64($0.0) &* $0.1 }
            for head in 0..<16 {
                let hash = head < 8 ? products[0] ^ products[1] : products[0] ^ products[1] ^ products[2]
                let modulus = vocabSizes[head]
                let value = ((hash % modulus) + modulus) % modulus + offsets[head]
                ids.append(UInt32(value))
            }
            prefix.append(token)
        }
        let embedding = tensor("per_layer_token_embd.weight").embedding(MLXArray(ids)).reshaped([1, tokens.count, 2560])
        let p = "blk.1.ple_"
        let keys = norm(linear(embedding, p + "key"), p + "norm_key", group: 2560).reshaped([1, x.dim(1), 4, 2560])
        let queries = norm(x, p + "norm_query", group: 2560).reshaped([1, x.dim(1), 4, 2560])
        let dot = sum(keys * queries, axis: -1, keepDims: true) / sqrt(Float(2560))
        let gate = sign(dot) * sqrt(maximum(abs(dot), 1e-6))
        let values = linear(embedding, p + "value").reshaped([1, x.dim(1), 1, 2560])
        let gated = (sigmoid(gate) * values).reshaped(x.shape)
        let normalized = norm(gated, p + "norm_conv", group: 2560)
        let input = concatenated([caches[1].pleConv ?? MLXArray.zeros([1, 9, 10240], dtype: x.dtype), normalized], axis: 1)
        caches[1].pleConv = input[0..., tokens.count..., 0...]
        if capturingPrefix { pleReplay.append(caches[1].pleConv!) }
        let weight = dense(p + "conv1d.weight").transposed()
        let windows = [0, 3, 6, 9].enumerated().map { j, start in input[0..., start..<(start+tokens.count), 0...] * weight[j] }
        let conv = sum(stacked(windows, axis: 2), axis: 2)
        return gated + silu(conv)
    }

    func draftForward(_ tokens: [Int], hidden: MLXArray, prefill: Bool = false) -> MLXArray {
        draftForward(MLXArray(tokens.map(UInt32.init)), count: tokens.count, hidden: hidden, prefill: prefill)
    }

    /// Draft attention needs a position count, not CPU token history. Keep token
    /// IDs on-device between speculative steps and synchronize candidates once.
    func draftForward(_ tokenIDs: MLXArray, count: Int, hidden: MLXArray, prefill: Bool = false) -> MLXArray {
        precondition(count > 0 && tokenIDs.size == count)
        if !prefill { clearPrefillCaches() }
        fastPrefill = prefill
        defer { fastPrefill = false }
        let length = count, prefix = "blk.48."
        let embedding = tensor("token_embd.weight").embedding(tokenIDs.asType(.uint32).reshaped([count])).reshaped([1, length, 2560])
        let projection = dense(prefix + "nextn.eh_proj.weight").asType(.float16)
        let embedded = matmul(norm(embedding, prefix + "nextn.enorm"), projection[0..., ..<2560].transposed())
        let normalized = norm(hidden, prefix + "nextn.hnorm").reshaped([1, length, 4, 2560])
        let projected = matmul(normalized, projection[0..., 2560...].transposed())
        var h = (projected + embedded.expandedDimensions(axis: 2)).reshaped([1, length, 10240])
        let (mixed, gates) = hyper(h, prefix + "hc_attn")
        h = inject(h, attention(mixed, 48), gates)
        let (ffn, injection) = hyper(h, prefix + "hc_ffn")
        h = inject(h, moe(ffn, prefix), injection)
        lastHidden = h[0..., (length-1)..., 0...]
        let (output, _) = hyper(lastHidden, "output_hc")
        draftPosition += count
        let logits = linear(output, "output").reshaped([-1])
        asyncEval(logits, lastHidden)
        return logits
    }

    struct Snapshot {
        let caches: [Cache]
        let history: [Int]
        let hidden: MLXArray
    }
    func snapshot() -> Snapshot { Snapshot(caches: caches, history: history, hidden: lastHidden) }
    func restore(_ snapshot: Snapshot) { caches = snapshot.caches; history = snapshot.history; lastHidden = snapshot.hidden }
    func commitVerifiedPrefix(from snapshot: Snapshot, count: Int) {
        let verifiedCount = history.count - snapshot.history.count
        precondition(count > 0 && count <= verifiedCount)
        defer { gdnReplay.removeAll(); pleReplay.removeAll() }
        guard count < verifiedCount else { return }
        let end = snapshot.history.count + count
        for i in caches.indices {
            if let inputs = gdnReplay[i] {
                let prefix = Self.gdnPrefixState(inputs, count: count, state: snapshot.caches[i].recurrent,
                                                 weights: gdnParameters[i]!)
                caches[i].conv = prefix[0]; caches[i].recurrent = prefix[1]
            }
            caches[i].keys = caches[i].keys?[0..., 0..., ..<end, 0...]
            caches[i].values = caches[i].values?[0..., 0..., ..<end, 0...]
            caches[i].indexKeys = caches[i].indexKeys?[0..., ..<end, 0...]
            caches[i].indexBlocks = caches[i].indexBlocks?[0..., 0..., ..<(end / 4), 0...]
        }
        precondition(gdnReplay.count == 36 && pleReplay.count == verifiedCount)
        caches[1].pleConv = pleReplay[count - 1]
        history = Array(history.prefix(end))
        lastHidden = lastHidden[0..., ..<count, 0...]
        asyncEval(caches.flatMap { [$0.conv, $0.recurrent, $0.pleConv].compactMap { $0 } })
    }

    func trimDraft(to count: Int) {
        precondition(count >= 0 && count <= draftPosition)
        caches[48].keys = caches[48].keys?[0..., 0..., ..<count, 0...]
        caches[48].values = caches[48].values?[0..., 0..., ..<count, 0...]
        caches[48].indexKeys = caches[48].indexKeys?[0..., ..<count, 0...]
        caches[48].indexBlocks = caches[48].indexBlocks?[0..., 0..., ..<(count / 4), 0...]
        draftPosition = count
    }

    func step(_ token: Int) -> MLXArray { forward([token]).reshaped([-1]) }

    func forward(_ tokens: [Int], prefill: Bool = false, lastLogitsOnly: Bool = false, capturePrefix: Bool = false) -> MLXArray {
        precondition(!tokens.isEmpty)
        precondition(!capturePrefix || (!prefill && tokens.count <= 8))
        gdnReplay.removeAll(); pleReplay.removeAll()
        capturingPrefix = capturePrefix
        if !prefill { clearPrefillCaches() }
        fastPrefill = prefill
        defer { fastPrefill = false; capturingPrefix = false }
        let embeddings = tensor("token_embd.weight").embedding(MLXArray(tokens.map(UInt32.init))).reshaped([1, tokens.count, 2560])
        let embedding: MLXArray
        if imageEmbeddings.isEmpty { embedding = embeddings }
        else {
            embedding = concatenated(tokens.indices.map { index in
                imageEmbeddings[history.count + index] ?? embeddings[0..., index..<(index+1), 0...]
            }, axis: 1)
        }
        var h = tiled(embedding, repetitions: [1, 1, 4])
        for i in 0..<48 {
            let p = "blk.\(i)."
            if i == 1 { h = h + ple(h, tokens: tokens) }
            trace?("\(i) input", h)
            let (mixed, gates) = hyper(h, p + "hc_attn")
            trace?("\(i) mixed", mixed)
            trace?("\(i) gates", gates)
            h = inject(h, (i+1) % 4 == 0 ? attention(mixed, i) : gdn(mixed, i), gates)
            trace?("\(i) attention", h)
            let (ffn, injection) = hyper(h, p + "hc_ffn")
            h = inject(h, moe(ffn, p), injection)
            trace?("\(i) moe", h)
            let stride = !prefill && tokens.count == 1 ? submissionStride : 1
            if (i + 1) % stride == 0 { asyncEval(h) }
        }
        lastHidden = h
        let (output, _) = hyper(lastLogitsOnly ? h[0..., (tokens.count-1)..., 0...] : h, "output_hc")
        history.append(contentsOf: tokens)
        let logits = linear(output, "output")
        // Submit state and logits together. The caller synchronizes when reading
        // the sampled token; waiting here would add a second GPU/CPU barrier.
        let pending = [logits, lastHidden] + caches.flatMap { [$0.conv, $0.recurrent, $0.indexKeys, $0.indexBlocks, $0.pleConv].compactMap { $0 } }
        if lastLogitsOnly {
            // Bound queued prefill work and retain chunk-level cancellation.
            eval(pending)
        } else {
            asyncEval(pending)
        }
        return logits
    }
}
