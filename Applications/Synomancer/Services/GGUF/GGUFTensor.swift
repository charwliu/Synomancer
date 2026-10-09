import Foundation
import MLX

/// Packed GGML blocks remain in their original quantization on the GPU.
final class GGUFTensor {
    var useCachedDispatch = true
    private var rowKernel: MLXFast.MLXFastKernel?
    private var matvecKernels: [Int: MLXFast.MLXFastKernel] = [:]
    private var batchKernels: [Int: MLXFast.MLXFastKernel] = [:]
    private var groupedKernels: [String: MLXFast.MLXFastKernel] = [:]
    private var prefillWeight: MLXArray?
    private var floatingWeight: MLXArray?
    private var floatingBatch: GGUFFloatingLinear?
    let info: GGUFFile.Tensor
    let packed: MLXArray
    let nativeGDNHeads: Bool
    let layout: (block: Int, bytes: Int, symbol: String)
    init(file: GGUFFile, tensor: GGUFFile.Tensor) throws {
        info = tensor
        layout = try GGUFFile.layout(tensor.type)
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
            packed = concatenated([loaded[..<4096], loaded[4096...]
                .reshaped([3, 16, 128, stride]).swappedAxes(0, 1).reshaped([6144, stride])], axis: 0)
            nativeGDNHeads = true
        } else if name.hasPrefix("blk."), tensor.dimensions.count == 2,
                  name.hasSuffix(".attn_gate.weight"), tensor.dimensions == [2560, 6144] {
            packed = loaded.reshaped([3, 16, 128, stride]).swappedAxes(0, 1).reshaped(shape)
            nativeGDNHeads = true
        } else if name.hasPrefix("blk."), tensor.dimensions == [2560, 48],
                  name.hasSuffix(".ssm_alpha.weight") || name.hasSuffix(".ssm_beta.weight") {
            packed = loaded.reshaped([3, 16, stride]).swappedAxes(0, 1).reshaped(shape)
            nativeGDNHeads = true
        } else if name.hasPrefix("blk."), tensor.dimensions == [6144, 2560],
                  name.hasSuffix(".ssm_out.weight"), ![UInt32(0), 1, 30].contains(tensor.type) {
            // Preserve the original reduction order. GEMV maps input indices
            // rather than permuting the quantized columns.
            packed = loaded
            nativeGDNHeads = true
        } else {
            packed = loaded
            nativeGDNHeads = false
        }
        eval(packed)
    }
    var width: Int { info.dimensions[0] }
    var rows: Int { info.dimensions.count > 1 ? info.dimensions[1] : 1 }

    func dense() -> MLXArray {
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
        if [UInt32(0), 1, 30].contains(info.type) { return take(dense(), ids, axis: 0).asType(.float16) }
        let stride = packed.dim(-1), width = width, block = layout.block, symbol = layout.symbol
        let kernel = rowKernel ?? MLXFast.metalKernel(name: "gguf_rows_\(symbol)", inputNames: ["packed", "ids"], outputNames: ["out"], source: """
            uint t = thread_position_in_grid.x;
            if (t >= ids_shape[0] * \(width / 16)) return;
            uint row = t / \(width / 16), col = t % \(width / 16);
            device const block_\(symbol) * blocks = (device const block_\(symbol) *)(packed + ulong(ids[row]) * \(stride));
            float4x4 value;
            dequantize_\(symbol)(blocks + col / \(block / 16), short(col % \(block / 16)), value);
            for (uint c = 0; c < 4; ++c) for (uint r = 0; r < 4; ++r) out[t*16+c*4+r] = value[c][r];
            """, header: GGUFMetalHeader.source)
        rowKernel = kernel
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
        if useCachedDispatch, let kernel = matvecKernels[repeats] {
            return kernel([packed, x, ids.asType(.uint32).reshaped([-1])],
                grid: (ids.size * rows * lanes, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [ids.shape + [rows]], outputDTypes: [x.dtype])[0]
        }
        let inputColumn = nativeGDNHeads && width == 6144
            ? "((col/8%16)*3+col/128)*128+(col%8)*16" : "col*16"
        let reduction = lanes == 32 ? "sum = simd_sum(sum);" : [4, 2, 1].map { "sum += simd_shuffle_down(sum, \($0));" }.joined(separator: "\n")
        let kernel = matvecKernels[repeats] ?? MLXFast.metalKernel(name: "gguf_gemv_\(symbol)", inputNames: ["packed", "x", "ids"], outputNames: ["out"], source: """
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
        matvecKernels[repeats] = kernel
        return kernel([packed, x, ids.asType(.uint32).reshaped([-1])], grid: (ids.size * rows * lanes, 1, 1), threadGroup: (128, 1, 1), outputShapes: [ids.shape + [rows]], outputDTypes: [x.dtype])[0]
    }
    private func batchLinear(_ x: MLXArray, count: Int) -> MLXArray {
        let width = width, rows = rows, stride = packed.dim(-1), symbol = layout.symbol
        let lanes = width <= 1024 ? 8 : 32
        if useCachedDispatch, let kernel = batchKernels[count] {
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
        let kernel = batchKernels[count] ?? MLXFast.metalKernel(name: "gguf_batch_\(symbol)_\(count)", inputNames: ["packed", "x"], outputNames: ["out"], source: """
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
        batchKernels[count] = kernel
        return kernel([packed, x], grid: (rows * lanes, 1, 1), threadGroup: (128, 1, 1), outputShapes: [Array(x.shape.dropLast()) + [rows]], outputDTypes: [x.dtype])[0]
    }

    private func grouped(_ x: MLXArray, ids: MLXArray, repeats: Int, routing: GGUFExpertRoutes) -> MLXArray {
        let experts = packed.dim(0), stride = packed.dim(-1), width = width, rows = rows
        let topK = routing.topK, selections = routing.selections, inputs = topK / repeats
        let key = "\(selections)/\(repeats)/\(topK)"
        let symbol = layout.symbol
        let kernel = groupedKernels[key] ?? MLXFast.metalKernel(name: "gguf_grouped_\(symbol)", inputNames: ["packed", "x", "counts", "routes"], outputNames: ["out"], source: """
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
        groupedKernels[key] = kernel
        return kernel([packed, x.asType(.float16), routing.counts, routing.routes], grid: (((routing.maximum + 31) / 32) * 128, (rows + 63) / 64, experts), threadGroup: (128, 1, 1), outputShapes: [ids.shape + [rows]], outputDTypes: [.float32])[0].asType(x.dtype)
    }

}
