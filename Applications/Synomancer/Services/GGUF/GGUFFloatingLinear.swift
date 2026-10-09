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
import MLX

/// Row-invariant floating GEMV adapted from mlx-vlm c244547 (MIT).
/// Matches singleton MLX GEMV reductions while evaluating verifier rows together.
final class GGUFFloatingLinear {
    private let kernel = MLXFast.metalKernel(
        name: "gguf_exact_floating_gemv", inputNames: ["x", "weight"], outputNames: ["out"],
        source: GGUFFloatingLinear.source, header: "#include <metal_simdgroup>\nusing namespace metal;\n")

    func callAsFunction(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let width = weight.dim(1), outputs = weight.dim(0), count = x.size / width
        precondition(x.dtype == .float16 && weight.dtype == x.dtype)
        precondition(outputs >= 4 && outputs % 4 == 0 && width < 16 * outputs)
        return kernel([x.reshaped([count, width]), weight],
            template: [("T", x.dtype), ("K", width), ("O", outputs), ("R", count)],
            grid: (32, outputs / 4, ((count + 7) / 8) * 8), threadGroup: (32, 1, 8),
            outputShapes: [[count, outputs]], outputDTypes: [x.dtype])[0]
            .reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    private static let source = #"""

        uint lane = thread_position_in_grid.x;
        uint out_block = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;

        constexpr int TM = 4;
        constexpr int TN = 4;
        constexpr int SN = 32;
        constexpr int blockN = SN * TN;

        if (row >= R) {
            return;
        }

        int out_row = int(out_block * TM);
        if (out_row >= O) {
            return;
        }

        const device T* in_vec = x + row * K;
        const device T* mat = weight + out_row * K;

        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        int col = int(lane * TN);
        int n_iter = K / blockN;
        int leftover = K - blockN * n_iter;

        for (int iter = 0; iter < n_iter; ++iter) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] = static_cast<float>(in_vec[col + tn]);
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    result[tm] +=
                        static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                }
            }

            col += blockN;
        }

        if (leftover > 0) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] =
                    (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    T m =
                        (col + tn < K) ? mat[tm * K + col + tn] : T(0);
                    result[tm] += static_cast<float>(m) * v[tn];
                }
            }
        }

        for (int tm = 0; tm < TM; ++tm) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }

        if (lane == 0) {
            for (int tm = 0; tm < TM; ++tm) {
                out[row * O + out_row + tm] = static_cast<T>(result[tm]);
            }
        }

    """#
}
