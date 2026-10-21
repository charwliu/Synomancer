import Foundation
import MLX

/// GGML blocks retain their original quantization, resident or read on demand.
final class GGUFTensor {
    var useCachedDispatch = true
    private final class Kernels {
        var rowKernel: MLXFast.MLXFastKernel?
        var matvecKernels: [Int: MLXFast.MLXFastKernel] = [:]
        var batchKernels: [Int: MLXFast.MLXFastKernel] = [:]
        var groupedKernels: [String: MLXFast.MLXFastKernel] = [:]
    }
    private var kernels = Kernels()
    private var prefillWeight: MLXArray?
    private var floatingWeight: MLXArray?
    private var floatingBatch: GGUFFloatingLinear?
    let info: GGUFFile.Tensor
    private let residentPacked: MLXArray?
    private let source: GGUFFile?
    private let byteCache: GGUFWeightCache?
    var isFileBacked: Bool { source != nil }
    var packed: MLXArray {
        precondition(residentPacked != nil, "File-backed weights require row/expert selection")
        return residentPacked!
    }
    var temporaryExpandedBytes: Int { prefillWeight?.nbytes ?? 0 }
    func clearPrefillCache() { prefillWeight = nil }
    let nativeGDNHeads: Bool
    let layout: (block: Int, bytes: Int, symbol: String)
    init(file: GGUFFile, tensor: GGUFFile.Tensor, cache: GGUFWeightCache? = nil) throws {
        info = tensor
        layout = try GGUFFile.layout(tensor.type)
        source = cache == nil ? nil : file
        byteCache = cache
        if cache != nil {
            residentPacked = nil; nativeGDNHeads = false
            return
        }
        let stride = tensor.dimensions[0] / layout.block * layout.bytes
        let shape = Array(tensor.shape.dropLast()) + [stride]
        let loaded = file.data.withUnsafeBytes { raw in
            MLXArray(UnsafeRawBufferPointer(rebasing: raw[tensor.offset..<(tensor.offset + tensor.byteCount)]), shape, type: UInt8.self)
        }
        // Restore GDN head layout once on packed rows/blocks, as the reference
        // loader does. This retains original quantization and avoids per-token
        // activation transposes. 128-wide heads must align with quant blocks.
        let name = tensor.name
        if name.hasPrefix("blk."), tensor.dimensions.count == 2,
           name.hasSuffix(".attn_qkv.weight"), tensor.dimensions == [2560, 10240] {
            residentPacked = concatenated([loaded[..<4096], loaded[4096...]
                .reshaped([3, 16, 128, stride]).swappedAxes(0, 1).reshaped([6144, stride])], axis: 0)
            nativeGDNHeads = true
        } else if name.hasPrefix("blk."), tensor.dimensions.count == 2,
                  name.hasSuffix(".attn_gate.weight"), tensor.dimensions == [2560, 6144] {
            residentPacked = loaded.reshaped([3, 16, 128, stride]).swappedAxes(0, 1).reshaped(shape)
            nativeGDNHeads = true
        } else if name.hasPrefix("blk."), tensor.dimensions == [2560, 48],
                  name.hasSuffix(".ssm_alpha.weight") || name.hasSuffix(".ssm_beta.weight") {
            residentPacked = loaded.reshaped([3, 16, stride]).swappedAxes(0, 1).reshaped(shape)
            nativeGDNHeads = true
        } else if name.hasPrefix("blk."), tensor.dimensions == [6144, 2560],
                  name.hasSuffix(".ssm_out.weight"), ![UInt32(0), 1, 30].contains(tensor.type) {
            // Preserve the original reduction order. GEMV maps input indices
            // rather than permuting the quantized columns.
            residentPacked = loaded
            nativeGDNHeads = true
        } else {
            residentPacked = loaded
            nativeGDNHeads = false
        }
        eval(packed)
    }
    private init(info: GGUFFile.Tensor, packed: MLXArray, layout: (block: Int, bytes: Int, symbol: String)) {
        self.info = info; self.residentPacked = packed; self.layout = layout
        source = nil; byteCache = nil; nativeGDNHeads = false
    }

    /// Read only selected records; evaluated packed buffers survive in the LRU.
    /// GPU concatenation avoids copying every cache hit through CPU staging.
    func selectedRecords(_ indices: [Int], expert: Bool) -> GGUFTensor {
        precondition(source != nil && !indices.isEmpty)
        let stride = width / layout.block * layout.bytes
        let recordBytes = stride * (expert ? rows : 1)
        let records = expert ? info.dimensions[2] : rows
        var recordsToPack: [MLXArray] = []
        recordsToPack.reserveCapacity(indices.count)
        for index in indices {
            precondition((0..<records).contains(index))
            let record = byteCache!.read(key: .init(tensor: info.name, index: index), byteCount: recordBytes) {
                let start = info.offset + index * recordBytes
                return source!.data.withUnsafeBytes { raw in
                    MLXArray(UnsafeRawBufferPointer(rebasing: raw[start..<(start + recordBytes)]),
                             expert ? [rows, stride] : [stride], type: UInt8.self)
                }
            }
            recordsToPack.append(record)
        }
        let dimensions = expert ? [width, rows, indices.count] : [width, indices.count]
        let array = stacked(recordsToPack, axis: 0)
        // Detach the compact staging buffer from cached/evicted source arrays.
        eval(array)
        let selected = GGUFTensor(info: .init(name: info.name, dimensions: dimensions, type: info.type,
            offset: 0, byteCount: indices.count * recordBytes), packed: array, layout: layout)
        selected.kernels = kernels
        return selected
    }

    var width: Int { info.dimensions[0] }
    var rows: Int { info.dimensions.count > 1 ? info.dimensions[1] : 1 }

    func dense() -> MLXArray {
        precondition(!isFileBacked, "Select file-backed rows before expanding weights")
        switch info.type {
        case 0: return packed.view(dtype: .float32).reshaped(info.shape)
        case 1: return packed.view(dtype: .float16).reshaped(info.shape)
        case 30: return packed.view(dtype: .bfloat16).reshaped(info.shape)
        default:
            let count = info.shape.dropLast().reduce(1, *)
            return embedding(MLXArray(0..<count).asType(.uint32)).reshaped(info.shape)
        }
    }

    func embedding(_ ids: MLXArray) -> MLXArray {
        if isFileBacked {
            let original = ids.asArray(Int.self)
            let unique = Array(Set(original)).sorted()
            let mapping = Dictionary(uniqueKeysWithValues: unique.enumerated().map { ($1, UInt32($0)) })
            let compact = MLXArray(original.map { mapping[$0]! }).reshaped(ids.shape)
            let result = selectedRecords(unique, expert: false).embedding(compact)
            eval(result)
            return result
        }
        if [UInt32(0), 1, 30].contains(info.type) { return take(dense(), ids, axis: 0).asType(.float16) }
        let stride = packed.dim(-1), width = width, block = layout.block, symbol = layout.symbol
        let kernel = kernels.rowKernel ?? MLXFast.metalKernel(name: "gguf_rows_\(symbol)", inputNames: ["packed", "ids"], outputNames: ["out"], source: """
            uint t = thread_position_in_grid.x;
            if (t >= ids_shape[0] * \(width / 16)) return;
            uint row = t / \(width / 16), col = t % \(width / 16);
            device const block_\(symbol) * blocks = (device const block_\(symbol) *)(packed + ulong(ids[row]) * \(stride));
            float4x4 value;
            dequantize_\(symbol)(blocks + col / \(block / 16), short(col % \(block / 16)), value);
            for (uint c = 0; c < 4; ++c) for (uint r = 0; r < 4; ++r) out[t*16+c*4+r] = value[c][r];
            """, header: GGUFMetalHeader.source)
        kernels.rowKernel = kernel
        return kernel([packed, ids.asType(.uint32).reshaped([-1])], grid: (ids.size * width / 16, 1, 1), threadGroup: (256, 1, 1), outputShapes: [ids.shape + [width]], outputDTypes: [.float16])[0]
    }

    func linear(_ x: MLXArray, prefill: Bool = false) -> MLXArray {
        let count = x.size / width
        if [UInt32(0), 1, 30].contains(info.type) {
            // BF16/F32 checkpoint matrices use the activation compute type. Keep
            // the converted matrix resident rather than recasting it each token.
            if floatingWeight == nil || floatingWeight!.dtype != x.dtype {
                floatingWeight = dense().asType(x.dtype).reshaped([rows, width])
                eval(floatingWeight!)
            }
            let weight = floatingWeight!.transposed()
            if (2...8).contains(count) {
                if x.dtype == .float16, rows >= 4, rows % 4 == 0, width < 16 * rows {
                    if floatingBatch == nil { floatingBatch = GGUFFloatingLinear() }
                    return floatingBatch!(x, weight: floatingWeight!)
                }
                let flat = x.reshaped([count, width])
                return concatenated((0..<count).map { row in
                    matmul(flat[row..<(row+1)], weight)
                }, axis: 0).reshaped(Array(x.shape.dropLast()) + [rows])
            }
            return matmul(x, weight).reshaped(Array(x.shape.dropLast()) + [rows])
        }
        if (2...8).contains(count) { return batchLinear(x, count: count) }
        if prefill && count > 8 {
            if prefillWeight == nil { prefillWeight = dense().asType(.float16); eval(prefillWeight!) }
            let input = nativeGDNHeads && width == 6144
                ? x.reshaped(Array(x.shape.dropLast()) + [16, 3, 128])
                    .swappedAxes(-3, -2).reshaped(x.shape) : x
            return matmul(input, prefillWeight!.transposed())
        }
        let ids = MLXArray.zeros([count], dtype: .uint32)
        return gathered(x, ids: ids, repeats: 1).reshaped(Array(x.shape.dropLast()) + [rows])
    }

    func gathered(_ x: MLXArray, ids: MLXArray, repeats: Int, routing: GGUFExpertRoutes? = nil) -> MLXArray {
        if let routing { return grouped(x, ids: ids, repeats: repeats, routing: routing) }
        let stride = packed.dim(-1), width = width, rows = rows, block = layout.block, symbol = layout.symbol
        let lanes = width <= 1024 ? 8 : 32
        if useCachedDispatch, let kernel = kernels.matvecKernels[repeats] {
            return kernel([packed, x, ids.asType(.uint32).reshaped([-1])],
                grid: (ids.size * rows * lanes, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [ids.shape + [rows]], outputDTypes: [x.dtype])[0]
        }
        let inputColumn = nativeGDNHeads && width == 6144
            ? "((col/8%16)*3+col/128)*128+(col%8)*16" : "col*16"
        let reduction = lanes == 32 ? "sum = simd_sum(sum);" : [4, 2, 1].map { "sum += simd_shuffle_down(sum, \($0));" }.joined(separator: "\n")
        let kernel = kernels.matvecKernels[repeats] ?? MLXFast.metalKernel(name: "gguf_gemv_\(symbol)", inputNames: ["packed", "x", "ids"], outputNames: ["out"], source: """
            uint t = thread_position_in_grid.x;
            uint output_row = t / \(lanes), lane = t % \(lanes), selection = output_row / \(rows);
            if (selection >= ids_shape[0]) return;
            uint row = output_row % \(rows);
            device const block_\(symbol) * blocks = (device const block_\(symbol) *)(packed + (ulong(ids[selection]) * \(rows) + row) * \(stride));
            ulong input_offset = ulong(selection / \(repeats)) * \(width);
            float sum = 0.0f;
            for (uint col = lane; col < \(width / 16); col += \(lanes)) {
                float4x4 value;
                dequantize_\(symbol)(blocks + col / \(block / 16), short(col % \(block / 16)), value);
                uint input_col = \(inputColumn);
                for (uint c = 0; c < 4; ++c) for (uint r = 0; r < 4; ++r)
                    sum += float(half(value[c][r])) * float(x[input_offset + input_col+c*4+r]);
            }
            \(reduction)
            if (lane == 0) out[output_row] = sum;
            """, header: GGUFMetalHeader.source)
        kernels.matvecKernels[repeats] = kernel
        return kernel([packed, x, ids.asType(.uint32).reshaped([-1])], grid: (ids.size * rows * lanes, 1, 1), threadGroup: (128, 1, 1), outputShapes: [ids.shape + [rows]], outputDTypes: [x.dtype])[0]
    }
    private func batchLinear(_ x: MLXArray, count: Int) -> MLXArray {
        let width = width, rows = rows, stride = packed.dim(-1), symbol = layout.symbol
        let lanes = width <= 1024 ? 8 : 32
        if useCachedDispatch, let kernel = kernels.batchKernels[count] {
            return kernel([packed, x], grid: (rows * lanes, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [Array(x.shape.dropLast()) + [rows]], outputDTypes: [x.dtype])[0]
        }
        let initialize = (0..<count).map { "float sum\($0) = 0.0f;" }.joined(separator: "\n")
        let inputColumn = nativeGDNHeads && width == 6144
            ? "((col/8%16)*3+col/128)*128+(col%8)*16" : "col*16"
        let accumulate = (0..<count).map { "sum\($0) += weight * float(x[\($0 * width) + input_col+c*4+r]);" }.joined(separator: "\n")
        let reduce = (0..<count).map { i in
            lanes == 32 ? "sum\(i) = simd_sum(sum\(i));" : [4, 2, 1].map { "sum\(i) += simd_shuffle_down(sum\(i), \($0));" }.joined(separator: "\n")
        }.joined(separator: "\n")
        let write = (0..<count).map { "out[\($0 * rows) + row] = sum\($0);" }.joined(separator: "\n")
        let kernel = kernels.batchKernels[count] ?? MLXFast.metalKernel(name: "gguf_batch_\(symbol)_\(count)", inputNames: ["packed", "x"], outputNames: ["out"], source: """
            uint t = thread_position_in_grid.x, row = t / \(lanes), lane = t % \(lanes);
            if (row >= \(rows)) return;
            device const block_\(symbol) * blocks = (device const block_\(symbol) *)(packed + ulong(row) * \(stride));
            \(initialize)
            for (uint col = lane; col < \(width / 16); col += \(lanes)) {
                float4x4 value;
                dequantize_\(symbol)(blocks + col / \(layout.block / 16), short(col % \(layout.block / 16)), value);
                uint input_col = \(inputColumn);
                for (uint c = 0; c < 4; ++c) for (uint r = 0; r < 4; ++r) {
                    float weight = float(half(value[c][r]));
                    \(accumulate)
                }
            }
            \(reduce)
            if (lane == 0) { \(write) }
            """, header: GGUFMetalHeader.source)
        kernels.batchKernels[count] = kernel
        return kernel([packed, x], grid: (rows * lanes, 1, 1), threadGroup: (128, 1, 1), outputShapes: [Array(x.shape.dropLast()) + [rows]], outputDTypes: [x.dtype])[0]
    }

    private func grouped(_ x: MLXArray, ids: MLXArray, repeats: Int, routing: GGUFExpertRoutes) -> MLXArray {
        let experts = packed.dim(0), stride = packed.dim(-1), width = width, rows = rows
        let topK = routing.topK, selections = routing.selections, inputs = topK / repeats
        let key = "\(experts)/\(selections)/\(repeats)/\(topK)"
        let symbol = layout.symbol
        let kernel = kernels.groupedKernels[key] ?? MLXFast.metalKernel(name: "gguf_grouped_\(symbol)", inputNames: ["packed", "x", "counts", "routes"], outputNames: ["out"], source: """
            threadgroup char shmem[8192];
            ggml_metal_kargs_mul_mm_id args = {
                \(width), \(experts), \(stride), \(rows * stride), \(experts * rows * stride),
                \(inputs), 2, \(width * 2), \(inputs * width * 2), 0,
                \(topK), \(selections), \(rows), \(topK), 1, 1
            };
            gguf_grouped_mm<half, half4x4, simdgroup_half8x8, half, half2x4,
                simdgroup_half8x8, block_\(symbol), \(layout.block / 16), dequantize_\(symbol),
                half, half4x4, half, half2x4>(args,
                (device const char *)packed, (device const char *)x,
                counts, (device const char *)routes, (device char *)out,
                (device const char *)packed, shmem, threadgroup_position_in_grid,
                thread_index_in_threadgroup, thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
            """, header: GGUFGroupedHeader.source)
        kernels.groupedKernels[key] = kernel
        return kernel([packed, x.asType(.float16), routing.counts, routing.routes], grid: (((routing.maximum + 31) / 32) * 128, (rows + 63) / 64, experts), threadGroup: (128, 1, 1), outputShapes: [ids.shape + [rows]], outputDTypes: [.float32])[0].asType(x.dtype)
    }

}
