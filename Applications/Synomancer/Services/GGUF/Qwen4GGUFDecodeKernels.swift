/*
MIT License

Copyright (c) 2026 yibie

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

/// Adapted from strata-mlx PR #9 (vendyluo), fused.py, MIT.
/// https://github.com/yibie/strata-mlx/pull/9
/// Preserve Synomancer's activation types, expert order and GDN normalization.
/// Only changed by the serial benchmark, between completed forwards. Model graph keys include this setting.
enum Qwen4GGUFDecodeKernels {
    // Bits: top-10 = 1, conv/SiLU = 2, gated norm = 4, MoE combine = 8.
    // M4 Max ablation: top-10, conv/SiLU and MoE combine help; gated norm is opt-in.
    static let defaultMask = 11
    nonisolated(unsafe) static var mask: Int = {
        if ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9"] == "0" { return 0 }
        return Int(ProcessInfo.processInfo.environment["SYNOMANCER_GGUF_PR9_MASK"] ?? "")
            .map { $0 & 15 } ?? defaultMask
    }()
    static var enabled: Bool {
        get { mask != 0 }
        set { mask = newValue ? defaultMask : 0 }
    }

    private static let topKernel = MLXFast.metalKernel(
        name: "gguf_decode_top10", inputNames: ["x"], outputNames: ["ids"], source: #"""
        uint row = threadgroup_position_in_grid.x, lane = thread_position_in_threadgroup.x;
        float values[16];
        for (uint j = 0; j < 16; ++j) values[j] = float(x[row * 512 + j * 32 + lane]);
        for (uint k = 0; k < 10; ++k) {
            float best = -INFINITY; uint index = 0;
            for (uint j = 0; j < 16; ++j) {
                uint at = j * 32 + lane;
                if (values[j] > best || (values[j] == best && at > index)) { best = values[j]; index = at; }
            }
            float value = simd_max(best);
            uint winner = simd_max(best == value ? index : 0u);
            // Existing argPartition takes the last ten in ascending order, ties by index.
            if (lane == 0) ids[row * 10 + 9 - k] = winner;
            if (winner % 32 == lane) values[winner / 32] = -INFINITY;
        }
        """#)

    static func top10(_ x: MLXArray) -> MLXArray {
        guard mask & 1 != 0, x.dim(-1) == 512, x.size / 512 <= 8 else {
            return argPartition(x, kth: -10, axis: -1)[.ellipsis, 502...]
        }
        let rows = x.size / 512
        return topKernel([x], template: [("T", x.dtype)], grid: (32 * rows, 1, 1),
                         threadGroup: (32, 1, 1), outputShapes: [Array(x.shape.dropLast()) + [10]],
                         outputDTypes: [.uint32])[0]
    }

    private static let gatedKernel = MLXFast.metalKernel(
        name: "gguf_decode_gated_norm", inputNames: ["x", "w", "z"], outputNames: ["out"], source: #"""
        #pragma clang fp contract(off)
        uint lane = thread_position_in_threadgroup.x, base = threadgroup_position_in_grid.x * 128 + lane * 4;
        float values[4], acc = 0;
        for (uint j = 0; j < 4; ++j) { values[j] = float(x[base+j]); acc += values[j] * values[j]; }
        float inv = metal::precise::rsqrt(simd_sum(acc) / 128.0f + 1e-6f);
        for (uint j = 0; j < 4; ++j) {
            W normalized = w[(base+j)%128] * W(values[j] * inv);
            float gate = float(z[base+j]);
            float s = 1.0f / (1.0f + metal::exp(metal::abs(gate)));
            out[base+j] = W(float(normalized) * (gate < 0 ? s : 1.0f-s));
        }
        """#)

    static func gatedNorm(_ x: MLXArray, weight: MLXArray, gate: MLXArray) -> MLXArray {
        guard mask & 4 != 0, x.dtype == .float16, [.float16, .float32].contains(weight.dtype), x.dim(-1) == 128,
              x.size / 128 <= 8 * 48 else {
            return Qwen4GGUFOps.floatSigmoidGate(MLXFast.rmsNorm(x, weight: weight, eps: 1e-6), gate)
        }
        return gatedKernel([x, weight, gate], template: [("T", x.dtype), ("W", weight.dtype)],
                           grid: (x.size / 4, 1, 1), threadGroup: (32, 1, 1),
                           outputShapes: [x.shape], outputDTypes: [weight.dtype])[0]
    }

    private static let convKernel = MLXFast.metalKernel(
        name: "gguf_decode_conv_silu", inputNames: ["state", "mixed", "weight"], outputNames: ["out", "joined"], source: #"""
        #pragma clang fp contract(off)
        uint c = thread_position_in_grid.x, t = thread_position_in_grid.y;
        if (c >= C) return;
        joined[(H+t)*C+c] = mixed[t*C+c];
        if (t == 0) for (uint j=0; j<H; ++j) joined[j*C+c] = state[j*C+c];
        float acc = 0;
        for (uint j=0; j<H+1; ++j) {
            uint row=t+j;
            T value = row < H ? state[row*C+c] : mixed[(row-H)*C+c];
            acc += float(value) * float(weight[c*(H+1)+j]);
        }
        T value=T(acc);
        T s=T(1)/(T(1)+metal::exp(metal::abs(value)));
        T sig=value < 0 ? s : T(1)-s;
        out[t*C+c]=value*sig;
        """#)

    static func convSilu(_ mixed: MLXArray, state: MLXArray, weight: MLXArray) -> (MLXArray, MLXArray)? {
        guard mask & 2 != 0, mixed.dim(1) <= 8, mixed.dtype == .float16, state.dtype == .float16,
              weight.dtype == .float16 else { return nil }
        let c = mixed.dim(2), h = state.dim(1), t = mixed.dim(1)
        let result = convKernel([state, mixed, weight], template: [("T", mixed.dtype), ("C", c), ("H", h)],
                                grid: (c, t, 1), threadGroup: (256, 1, 1),
                                outputShapes: [mixed.shape, [1, h+t, c]], outputDTypes: [mixed.dtype, mixed.dtype])
        return (result[0], result[1])
    }
    private static let combineKernel = MLXFast.metalKernel(
        name: "gguf_decode_moe_combine", inputNames: ["y", "w", "g", "s"], outputNames: ["out"], source: #"""
        #pragma clang fp contract(off)
        uint i=thread_position_in_grid.x, t=thread_position_in_grid.y;
        if (i >= D) return;
        T acc=0;
        for (uint j=0; j<8; ++j) {
            T part=0;
            for (uint k=j; k<10; k+=8) part=T(y[(t*10+k)*D+i]*w[t*10+k])+part;
            acc=j==0 ? part : part+acc;
        }
        T v=g[t];
        T sig=T(1)/(T(1)+metal::exp(metal::abs(v)));
        T gate=v<0 ? sig : T(1)-sig;
        out[t*D+i]=T(acc)+T(gate*s[t*D+i]);
        """#)

    static func combine(_ y: MLXArray, scores: MLXArray, gate: MLXArray, shared: MLXArray) -> MLXArray {
        guard mask & 8 != 0, y.dtype == .float16, scores.dtype == .float16,
              y.dim(-2) == 10, shared.size / shared.dim(-1) <= 8 else {
            return sum(y * scores.expandedDimensions(axis: -1), axis: -2) + sigmoid(gate) * shared
        }
        let d=shared.dim(-1), rows=shared.size/d
        return combineKernel([y, scores, gate, shared], template: [("T", y.dtype), ("D", d)],
                             grid: (d, rows, 1), threadGroup: (256, 1, 1),
                             outputShapes: [shared.shape], outputDTypes: [shared.dtype])[0]
    }

}
