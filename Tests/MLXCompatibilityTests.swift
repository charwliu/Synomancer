import Foundation
import MLX
import MLXNN
import ComplexModule

@main
struct MLXCompatibilityTests {
    static func main() {
        let floatSorted = sorted(MLXArray([Float.nan, 3, -1, Float.nan, 2])).asArray(Float.self)
        precondition(Array(floatSorted.prefix(3)) == [-1, 2, 3] && floatSorted.suffix(2).allSatisfy(\.isNaN))
        precondition(sorted(MLXArray([3, -1, 2])).asArray(Int.self) == [-1, 2, 3])
        let complex = stacked([MLXArray(Complex<Float>(2, 0)), MLXArray(Complex<Float>(-1, 0)), MLXArray(Complex<Float>(.nan, 0))])
        let complexSorted = sorted(complex)
        precondition(complexSorted[0].item(Complex<Float>.self).real == -1)
        precondition(complexSorted[2].item(Complex<Float>.self).real.isNaN)

        let input = MLXArray([Float(1), 2, 3, 4])
        let normalized = MLXFast.layerNorm(input, eps: 1e-5)
        precondition(abs(normalized.mean().item(Float.self)) < 1e-5)
        precondition(abs(normalized.variance().item(Float.self) - 1) < 1e-4)

        for dimension in [64, 128] {
            let q = ones([1, 1, 4, dimension])
            let v = arange(4 * dimension).asType(.float32).reshaped(1, 1, 4, dimension)
            let mask = MLXArray([true, true, false, false]).reshaped(1, 1, 1, 4)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: q, values: v, scale: 1 / sqrt(Float(dimension)),
                mask: mask, forceFused: true)
            let expected = (v[0, 0, 0] + v[0, 0, 1]) / 2
            precondition(abs(attended - expected).max().item(Float.self) < 1e-3)
        }

        let layer = SwitchLinear(
            inputDims: 64, outputDims: 4, numExperts: 2,
            weight: arange(512).asType(.float32).reshaped(2, 4, 64) / 512)
        let x = ones([1, 64])
        let indices = MLXArray([0, 1])
        let reference = layer(x, indices)
        let quantized = layer.toQuantized(groupSize: 64, bits: 4, mode: .affine) as! QuantizedSwitchLinear
        precondition(quantized.mode == .affine && quantized.biases != nil)
        precondition(abs(reference - quantized(x, indices)).max().item(Float.self) < 1)
        let symmetric = layer.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4) as! QuantizedSwitchLinear
        precondition(symmetric.biases == nil)
        precondition(abs(reference - symmetric(x, indices)).max().item(Float.self) < 8)
        Stream.defaultStream(.gpu).synchronize()
        print("MLX GPU regressions passed: integer/float/complex sorting, normalization, masked attention and affine/MXFP4 experts")
    }
}
