import Foundation
import MLX
import MLXNN

/// Synthetic blocks, not distributed model weights. Compares independent GEMV/GEMM
/// dispatch paths, including duplicate expert IDs and partial 32-route tiles.
enum GGUFKernelTests {
    static func run() throws {
        let probe = MLXArray.ones([1, 1, 16, 128], dtype: .float16)
        print("GDN normalized Q/K dtype", Qwen4GGUFOps.normalizeQK([probe, probe])[0].dtype)
        let originalMask = Qwen4GGUFDecodeKernels.mask
        Qwen4GGUFDecodeKernels.mask = 15
        defer { Qwen4GGUFDecodeKernels.mask = originalMask }
        for length in [1, 2, 4, 8, 16] {
            for ties in [false, true] {
                let values = sin(MLXArray(0..<(length * 512)).asType(.float32) / 17)
                let x = (ties ? (values * 3).asType(.int32).asType(.float16) : values.asType(.float16)).reshaped([1, length, 512])
                let expected = argPartition(x, kth: -10, axis: -1)[.ellipsis, 502...]
                let got = Qwen4GGUFDecodeKernels.top10(x)
                precondition(all(got .== expected).item(Bool.self), "PR9 top-k order/ties changed")
            }
            let x = sin(MLXArray(0..<(length * 48 * 128)).asType(.float32) / 19).asType(.float16).reshaped([1, length, 48, 128])
            let z = (cos(x.asType(.float32) * 13) * 12).asType(.float16)
            let w = cos(MLXArray(0..<128).asType(.float32) / 31).asType(.float16)
            let expected = Qwen4GGUFOps.floatSigmoidGate(MLXFast.rmsNorm(x, weight: w, eps: 1e-6), z)
            let got = Qwen4GGUFDecodeKernels.gatedNorm(x, weight: w, gate: z)
            print("GATED", length, max(abs(got.asType(.float32)-expected.asType(.float32))).item(Float.self))
            precondition(all(got .== expected).item(Bool.self), "PR9 gated norm changed rounding")
            let shared = x.reshaped([1,length,6144])[.ellipsis, ..<2560]
            let expert = sin(MLXArray(0..<(length*10*2560)).asType(.float32)/23).asType(.float16).reshaped([1,length,10,2560])
            let scores = softmax(cos(MLXArray(0..<(length*10)).asType(.float32)).asType(.float16).reshaped([1,length,10]), axis: -1)
            let gate = shared[.ellipsis, ..<1]
            let reference = sum(expert*scores.expandedDimensions(axis: -1), axis: -2)+sigmoid(gate)*shared
            let combined = Qwen4GGUFDecodeKernels.combine(expert, scores: scores, gate: gate, shared: shared)
            print("COMBINE", length, max(abs(combined.asType(.float32)-reference.asType(.float32))).item(Float.self))
            precondition(all(combined .== reference).item(Bool.self), "PR9 MoE combination changed rounding")
            let floatWeight = w.asType(.float32) + 0.0001
            let floatExpected = Qwen4GGUFOps.floatSigmoidGate(MLXFast.rmsNorm(x, weight: floatWeight, eps: 1e-6), z)
            let floatGot = Qwen4GGUFDecodeKernels.gatedNorm(x, weight: floatWeight, gate: z)
            print("GATED F32", length, max(abs(floatGot-floatExpected)).item(Float.self))
            precondition(all(floatGot .== floatExpected).item(Bool.self), "PR9 F32 weight normalization changed")
            let channels = 10240
            let input = sin(MLXArray(0..<((length+3)*channels)).asType(.float32) / 31).asType(.float16).reshaped([1,length+3,channels])
            let weight = cos(MLXArray(0..<(channels*4)).asType(.float32) / 19).asType(.float16).reshaped([channels,4,1])
            if let (out, joined) = Qwen4GGUFDecodeKernels.convSilu(input[0...,3...,0...], state: input[0...,..<3,0...], weight: weight) {
                let expected = silu(conv1d(input, weight, groups: channels))
                print("CONV", length, max(abs(out.asType(.float32)-expected.asType(.float32))).item(Float.self))
                precondition(all(out .== expected).item(Bool.self), "PR9 conv/silu rounding changed")
                precondition(all(joined .== input).item(Bool.self), "PR9 convolution history changed")
            }
        }
        print("PASS: PR9 top-k ties/order, gated norm, conv/silu/history and fallback")
        precondition(Qwen4GGUFModel.prefillRanges(count: 1, step: 16, separateLast: true) == [0..<1])
        precondition(Qwen4GGUFModel.prefillRanges(count: 32, step: 16, separateLast: true) == [0..<16, 16..<31, 31..<32])
        precondition(Qwen4GGUFModel.prefillRanges(count: 33, step: 16, separateLast: true) == [0..<16, 16..<32, 32..<33])
        precondition(Qwen4GGUFModel.prefillRanges(count: 32, step: 16, separateLast: false) == [0..<16, 16..<32])

        let prefixes = Array(repeating: 1, count: 13) + Array(repeating: 0, count: 7)
        precondition(Qwen4GGUFMTP.adaptiveDraftCount(round: 20, warmupRounds: 8, acceptances: prefixes) == 3,
                     "Accepted first drafts must count as prefix hits even when later drafts fail")
        precondition(Qwen4GGUFMTP.adaptiveDraftCount(round: 20, warmupRounds: 8, acceptances: Array(prefixes.dropFirst())) == 1,
                     "Below 65 percent prefix hits must retain the trained one-token depth")
        precondition(Qwen4GGUFMTP.adaptiveDraftCount(round: 64, warmupRounds: 8,
                     acceptances: Array(repeating: 0, count: 32) + Array(repeating: 1, count: 32)) == 3,
                     "Old misses must not override the recent 32-round window")
        precondition(Qwen4GGUFMTP.adaptiveDraftCount(round: 7, warmupRounds: 8, acceptances: [1, 1, 1]) == 1)
        print("PASS: MTP prefix-hit policy, warmup, 65 percent boundary and rolling window")

        for length in [1, 4, 512] {
            let positions = MLXArray(0..<(length * 128)).asType(.float32)
            let value = (sin(positions / 19) * 8).asType(.float16).reshaped([1, length, 128])
            let gate = (cos(positions / 31) * 12).asType(.float16).reshaped(value.shape)
            let expected = value * sigmoid(gate)
            let expectedFloat = (value.asType(.float32) * sigmoid(gate.asType(.float32))).asType(.float16)
            precondition(all(Qwen4GGUFOps.sigmoidGate(value, gate) .== expected).item(Bool.self), "Fused attention gate changed rounding")
            precondition(all(Qwen4GGUFOps.floatSigmoidGate(value, gate) .== expectedFloat).item(Bool.self), "Fused GDN gate changed rounding")
        }
        print("PASS: fused attention/GDN gates preserve rounding")
        let floating = GGUFFloatingLinear()
        for (width, outputs) in [(256, 64), (2560, 512), (512, 10240), (10240, 1024)] {
            let weight = cos(MLXArray(0..<(width * outputs)).asType(.float32) / 73)
                .asType(.float16).reshaped([outputs, width])
            for count in [2, 4, 8] {
                let input = sin(MLXArray(0..<(count * width)).asType(.float32) / 31)
                    .asType(.float16).reshaped([1, count, width])
                let expected = concatenated((0..<count).map {
                    matmul(input[0..., $0..<($0+1), 0...], weight.transposed())
                }, axis: 1)
                precondition(all(floating(input, weight: weight) .== expected).item(Bool.self),
                             "Floating verifier GEMV differs from singleton \(width)x\(outputs)")
            }
        }
        print("PASS: floating verifier GEMV preserves singleton reductions at model dimensions")
        // Native grouped convolution must preserve verifier row invariance.
        let channels = 10240
        let weight = cos(MLXArray(0..<(channels * 4)).asType(.float32) / 19)
            .asType(.float16).reshaped([channels, 4, 1])
        for length in [2, 4, 8, 512] {
            let input = sin(MLXArray(0..<((length + 3) * channels)).asType(.float32) / 31)
                .asType(.float16).reshaped([1, length + 3, channels])
            let batched = conv1d(input, weight, groups: channels)
            let single = concatenated((0..<length).map { row in
                conv1d(input[0..., row..<(row+4), 0...], weight, groups: channels)
            }, axis: 1)
            precondition(all(batched .== single).item(Bool.self), "Grouped convolution is not row invariant")
        }
        print("PASS: grouped convolution verifier rows and prefill are singleton-exact")
        let sparse = Qwen4GGUFSparseAttention()
        for (length, total) in [(8, 8), (4, 2060)] {
            let q = sin(MLXArray(0..<(24 * length * 256)).asType(.float32) / 13)
                .asType(.float16).reshaped([1, 24, length, 256])
            let k = cos(MLXArray(0..<(2 * total * 256)).asType(.float32) / 17)
                .asType(.float16).reshaped([1, 2, total, 256])
            let v = sin(k.asType(.float32) * 3).asType(.float16)
            let ends = MLXArray((total-length+1)...total).reshaped([1, length])
            let complete = (ends / 4).asType(.int32).expandedDimensions(axis: -1)
            let blockIDs = MLXArray(0..<512).reshaped([1, 1, 512])
            let selected = which(blockIDs .< complete, blockIDs, -1)
            let output = sparse(q, k, v, blocks: selected, queryEnds: ends)
            let singleton = concatenated((0..<length).map { row in
                let end = total - length + row + 1
                return sparse(q[0..., 0..., row..<(row+1), 0...],
                    k[0..., 0..., ..<end, 0...], v[0..., 0..., ..<end, 0...],
                    blocks: selected[0..., row..<(row+1), 0...], queryEnds: ends[0..., row..<(row+1)])
            }, axis: 2)
            precondition(all(output .== singleton).item(Bool.self), "QSA verifier is not row invariant")
            let indices = MLXArray(0..<total).reshaped([1, 1, total])
            let visible = indices .< ends.expandedDimensions(axis: -1)
            let mask = visible .&& ((indices .< 2048) .|| (indices .>= complete * 4))
            let dense = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / 16,
                mask: mask.reshaped([1, 1, length, total]))
            precondition(allClose(output, dense, rtol: 0.01, atol: 0.005).item(Bool.self), "QSA indexed/masked attention mismatch")
        }
        print("PASS: indexed QSA singleton equality, causal tails, sentinel blocks and masked SDPA comparison")
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        func u32(_ n: UInt32) -> Data { var v = n.littleEndian; return withUnsafeBytes(of: &v) { Data($0) } }
        func u64(_ n: UInt64) -> Data { var v = n.littleEndian; return withUnsafeBytes(of: &v) { Data($0) } }
        func fixture(type: UInt32, experts: Int, width: Int = 256, rows: Int = 64, name: String = "weight", cache: GGUFWeightCache? = nil) throws -> GGUFTensor {
            let layout = try GGUFFile.layout(type)
            let stride = width / layout.block * layout.bytes
            var data = u32(0x46554747) + u32(3) + u64(1) + u64(0)
            data += u64(UInt64(name.utf8.count)) + Data(name.utf8) + u32(experts == 1 ? 2 : 3) + u64(UInt64(width)) + u64(UInt64(rows))
            if experts > 1 { data += u64(UInt64(experts)) }
            data += u32(type) + u64(0)
            while data.count % 32 != 0 { data.append(0) }
            if [UInt32(0), 1, 30].contains(type) {
                for index in 0..<(experts * rows * width) {
                    let value = sin(Float(index) / 19)
                    if type == 0 { data += u32(value.bitPattern) }
                    else {
                        var bits = (type == 1 ? Float16(value).bitPattern : UInt16(value.bitPattern >> 16)).littleEndian
                        data += withUnsafeBytes(of: &bits) { Data($0) }
                    }
                }
            } else {
                data += Data((0..<(experts * rows * stride)).map { index in
                    let row = index / stride
                    let salt = row ^ (row >> 3) ^ (row >> 7)
                    return UInt8(0x11 + (index * 7 + salt) % 8)
                })
            }
            let path = directory.appendingPathComponent("\(UUID().uuidString).gguf")
            try data.write(to: path)
            let file = try GGUFFile(url: path)
            return try GGUFTensor(file: file, tensor: file.tensors[name]!, cache: cache)
        }
        let lru = GGUFWeightCache(budget: 8)
        func read(_ index: Int, _ count: Int = 4) -> MLXArray {
            lru.read(key: .init(tensor: "test", index: index), byteCount: count) { MLXArray(Array(repeating: UInt8(index), count: count)) }
        }
        _ = read(1); _ = read(2); _ = read(1); _ = read(3)
        precondition(lru.bytes == 8 && lru.hits == 1 && lru.misses == 3)
        _ = read(1); _ = read(2)
        precondition(lru.hits == 2 && lru.misses == 4, "LRU must evict the least recently used record")
        _ = read(9, 16)
        precondition(lru.bytes == 8, "Oversize record must bypass the cache")
        lru.removeAll(); precondition(lru.bytes == 0)
        let entryLimited = GGUFWeightCache(budget: 1024, maximumEntries: 2)
        for index in 0..<3 {
            _ = entryLimited.read(key: .init(tensor: "row", index: index), byteCount: 1) { MLXArray([UInt8(index)]) }
        }
        precondition(entryLimited.bytes == 2, "Tiny PLE buffers must also obey the entry-count cap")
        for type: UInt32 in [0, 1, 30, 8, 21] {
            let resident = try fixture(type: type, experts: 1)
            let cache = GGUFWeightCache(budget: 2048)
            let paged = try fixture(type: type, experts: 1, cache: cache)
            for values: [UInt32] in [[63, 0, 63, 17], [17, 1, 2, 63]] {
                let ids = MLXArray(values).reshaped([2, 2])
                precondition(all(resident.embedding(ids) .== paged.embedding(ids)).item(Bool.self),
                    "Paged embedding differs for type \(type)")
                precondition(cache.bytes <= cache.budget)
            }
        }
        for type: UInt32 in [8, 21] {
            let resident = try fixture(type: type, experts: 4)
            let cache = GGUFWeightCache(budget: 0)
            let paged = try fixture(type: type, experts: 4, cache: cache)
            let selected = paged.selectedRecords([1, 3], expert: true)
            for length in [1, 4, 129] {
                let x = sin(MLXArray(0..<(length * 256)).asType(.float32) / 19)
                    .asType(.float16).reshaped([1, length, 256])
                let original = MLXArray(Array(repeating: [UInt32(3), 1, 3], count: length).flatMap { $0 }).reshaped([1, length, 3])
                let compact = MLXArray(Array(repeating: [UInt32(1), 0, 1], count: length).flatMap { $0 }).reshaped(original.shape)
                let a = resident.gathered(x, ids: original, repeats: 3,
                    routing: length > 8 ? GGUFExpertRoutes(ids: original, experts: 4) : nil)
                let b = selected.gathered(x, ids: compact, repeats: 3,
                    routing: length > 8 ? GGUFExpertRoutes(ids: compact, experts: 2) : nil)
                precondition(all(a .== b).item(Bool.self), "Paged expert routing changed output")
            }
            precondition(cache.bytes == 0 && cache.misses == 2)
        }
        let expanded = try fixture(type: 8, experts: 1)
        let expandedInput = MLXArray.ones([1, 16, 256], dtype: .float16)
        let expandedOutput = expanded.linear(expandedInput, prefill: true)
        eval(expandedOutput)
        precondition(expanded.temporaryExpandedBytes > 0)
        expanded.clearPrefillCache()
        precondition(expanded.temporaryExpandedBytes == 0)
        precondition(all(expandedOutput .== expanded.linear(expandedInput, prefill: true)).item(Bool.self))
        print("PASS: paged PLE rows, duplicate IDs, expert GEMV/grouped remapping, zero-budget/LRU eviction and expanded-cache release")
        let gdnWeights = try Qwen4GGUFModel.GDNGraphWeights(
            qkv: fixture(type: 8, experts: 1, width: 2560, rows: 10240, name: "blk.0.attn_qkv.weight"),
            gate: fixture(type: 8, experts: 1, width: 2560, rows: 6144, name: "blk.0.attn_gate.weight"),
            alpha: fixture(type: 30, experts: 1, width: 2560, rows: 48, name: "blk.0.ssm_alpha.weight"),
            beta: fixture(type: 30, experts: 1, width: 2560, rows: 48, name: "blk.0.ssm_beta.weight"),
            projection: fixture(type: 8, experts: 1, width: 6144, rows: 2560, name: "blk.0.ssm_out.weight"),
            convolution: (cos(MLXArray(0..<40960).asType(.float32) / 43) / 4).asType(.float16).reshaped([10240, 4, 1]),
            aLog: sin(MLXArray(0..<48).asType(.float32)) / 5,
            bias: cos(MLXArray(0..<48).asType(.float32) / 7) / 4,
            norm: 1 + sin(MLXArray(0..<128).asType(.float32) / 19) / 10)
        var convState = MLXArray.zeros([1, 3, 10240], dtype: .float16)
        var recurrent = (sin(MLXArray(0..<(48 * 128 * 128)).asType(.float32) / 59) / 100)
            .reshaped([1, 48, 128, 128])
        let gdnGraph = compile { (inputs: [MLXArray]) in
            Qwen4GGUFModel.gdnOutput(inputs[0], conv: inputs[1], state: inputs[2], weights: gdnWeights, prefill: false)
        }
        for (iteration, count) in [1, 4, 1].enumerated() {
            let x = sin(MLXArray(0..<(count * 2560)).asType(.float32) / Float(31 + iteration))
                .asType(.float16).reshaped([1, count, 2560])
            let oldState = recurrent.asArray(Float.self), oldConv = convState.asArray(Float.self)
            let expected = Qwen4GGUFModel.gdnOutput(x, conv: convState, state: recurrent, weights: gdnWeights, prefill: false)
            eval(expected)
            let actual = gdnGraph([x, convState, recurrent])
            precondition(zip(actual, expected).allSatisfy { all($0 .== $1).item(Bool.self) }, "Compiled GDN changed outputs/state")
            precondition(recurrent.asArray(Float.self) == oldState && convState.asArray(Float.self) == oldConv,
                         "Compiled GDN mutated a snapshot input")
            if count == 4 {
                let captured = Qwen4GGUFModel.gdnOutput(x, conv: convState, state: recurrent,
                                                      weights: gdnWeights, prefill: false, captureInputs: true)
                for prefix in 1..<count {
                    let expectedPrefix = Qwen4GGUFModel.gdnOutput(x[0..., ..<prefix, 0...], conv: convState,
                        state: recurrent, weights: gdnWeights, prefill: false)
                    let restored = Qwen4GGUFModel.gdnPrefixState(Array(captured.dropFirst(3)), count: prefix,
                                                               state: recurrent, weights: gdnWeights)
                    precondition(all(restored[0] .== expectedPrefix[1]).item(Bool.self)
                                 && all(restored[1] .== expectedPrefix[2]).item(Bool.self),
                                 "GDN prefix replay changed convolution/recurrent state")
                }
                precondition(recurrent.asArray(Float.self) == oldState && convState.asArray(Float.self) == oldConv,
                             "GDN prefix replay mutated its original snapshot")
            }
            convState = expected[1]; recurrent = expected[2]
        }
        print("PASS: compiled GDN preserves explicit recurrence, convolution state and immutable snapshot inputs")
        let moeWeights = try [
            fixture(type: 1, experts: 1, width: 1280, rows: 512),
            fixture(type: 8, experts: 512, width: 1280),
            fixture(type: 14, experts: 512, width: 1280),
            fixture(type: 8, experts: 512, width: 64, rows: 1280),
            fixture(type: 8, experts: 1, width: 1280),
            fixture(type: 14, experts: 1, width: 1280),
            fixture(type: 8, experts: 1, width: 64, rows: 1280),
            fixture(type: 1, experts: 1, width: 1280, rows: 1)
        ]
        let compiledMoE = compile { x in Qwen4GGUFModel.experts(x, weights: moeWeights, prefill: false) }
        for (iteration, count) in [1, 4, 8, 1].enumerated() {
            let x = sin(MLXArray(0..<(count * 1280)).asType(.float32) / Float(31 + iteration))
                .asType(.float16).reshaped([1, count, 1280])
            let reference = Qwen4GGUFModel.experts(x, weights: moeWeights, prefill: false)
            eval(reference)
            precondition(all(compiledMoE(x) .== reference).item(Bool.self), "Compiled MoE changed outputs or routes")
        }
        print("PASS: compiled MoE output equality across verifier shapes and changing routes")
        for (suffix, rows, headSize) in [("attn_qkv", 10240, 128), ("attn_gate", 6144, 128), ("ssm_alpha", 48, 1), ("ssm_beta", 48, 1)] {
            let raw = try fixture(type: 8, experts: 1, width: 2560, rows: rows)
            let restored = try fixture(type: 8, experts: 1, width: 2560, rows: rows, name: "blk.0.\(suffix).weight")
            let x = sin(MLXArray(0..<5120).asType(.float32) / 23).asType(.float16).reshaped([1, 2, 2560])
            let original = raw.linear(x)
            let tail = suffix == "attn_qkv" ? original[.ellipsis, 4096...] : original
            let permuted = tail.reshaped([1, 2, 3, 16, headSize]).swappedAxes(2, 3).reshaped([1, 2, 48 * headSize])
            let expected = suffix == "attn_qkv" ? concatenated([original[.ellipsis, ..<4096], permuted], axis: -1) : permuted
            precondition(all(restored.linear(x) .== expected).item(Bool.self), "Packed GDN row restoration mismatch")
        }
        let originalProjection = try fixture(type: 8, experts: 1, width: 6144, rows: 2560)
        let restoredProjection = try fixture(type: 8, experts: 1, width: 6144, rows: 2560, name: "blk.0.ssm_out.weight")
        for count in [1, 2, 4, 16] {
            let x = sin(MLXArray(0..<(count * 6144)).asType(.float32) / 29).asType(.float16).reshaped([1, count, 6144])
            let packedOrder = x.reshaped([1, count, 16, 3, 128]).swappedAxes(2, 3).reshaped(x.shape)
            let expected = originalProjection.linear(packedOrder, prefill: count > 8)
            precondition(all(restoredProjection.linear(x, prefill: count > 8) .== expected).item(Bool.self), "GDN projection changed reduction order")
        }
        print("PASS: packed GDN head restoration and indexed output projection preserve exact reductions")
        if ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_MICROBENCH"] == "1" {
            for count in [1, 4] {
                let x = sin(MLXArray(0..<(count * 6144)).asType(.float32) / 29).asType(.float16).reshaped([1, count, 6144])
                let packedOrder = x.reshaped([1, count, 16, 3, 128]).swappedAxes(2, 3).reshaped(x.shape)
                eval(x, packedOrder)
                for cached in [false, true] {
                    originalProjection.useCachedDispatch = cached
                    restoredProjection.useCachedDispatch = cached
                    for (label, weight, input) in [("original", originalProjection, packedOrder), ("indexed", restoredProjection, x)] {
                        for _ in 0..<5 { eval(weight.linear(input)) }
                        let start = Date()
                        for _ in 0..<100 { eval(weight.linear(input)) }
                        print("MICRO", label, "count", count, "cached", cached,
                              "milliseconds", Date().timeIntervalSince(start) * 10)
                    }
                }
            }
        }
        for type in [UInt32(0), 1, 30, 8, 14, 13, 12, 42, 23, 20, 21, 18, 22] {
            for width in [256, 2560, 10240] {
                let linear = try fixture(type: type, experts: 1, width: width)
                for count in [2, 4, 8] {
                    let values = sin(MLXArray(0..<(count * width)).asType(.float32) / 31)
                    let x = values.asType(.float16).reshaped([1, count, width])
                    let batched = linear.linear(x)
                    let single = concatenated((0..<count).map { linear.linear(x[0..., $0..<($0+1), 0...]) }, axis: 1)
                    precondition(all(batched .== single).item(Bool.self), "Small batch reduction changed type \(type), width \(width)")
                }
            }
            if [UInt32(0), 1, 30].contains(type) {
                print("PASS: small-batch exact floating GEMV", type)
                continue
            }
            let experts = try fixture(type: type, experts: 8)
            let routeIDs: [UInt32] = (0..<1290).map { index in UInt32((index * 3 + index / 10) % 8) }
            let ids = MLXArray(routeIDs).reshaped([1, 129, 10])
            let plan = GGUFExpertRoutes(ids: ids, experts: 8)
            for repeats in [1, 10] {
                let shape = repeats == 1 ? [1, 129, 10, 256] : [1, 129, 256]
                let count = shape.reduce(1, *)
                let x = sin(MLXArray(0..<count).asType(.float32) / 31).asType(.float16).reshaped(shape)
                let gemv = experts.gathered(x, ids: ids, repeats: repeats)
                let gemm = experts.gathered(x, ids: ids, repeats: repeats, routing: plan)
                precondition(allClose(gemv, gemm, rtol: 0.003, atol: 0.003).item(Bool.self), "Grouped routing mismatch \(type), repeats \(repeats)")
            }
            print("PASS: small-batch exact GEMV and grouped GEMM", try GGUFFile.layout(type).symbol)
        }
    }
}
