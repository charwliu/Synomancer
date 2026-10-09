import Foundation
import MLX
import MLXNN

/// Fused pointwise graphs shared across layers; weights and recurrent state stay
/// explicit inputs. Matches the native Qwen4 equations used by mlx-vlm.
enum Qwen4GGUFOps {
    static let norm = compile { (x: MLXArray, weight: MLXArray) in
        let f = x.asType(.float32)
        return (f * rsqrt(mean(square(f), axis: -1, keepDims: true) + 1e-6)
            * weight.asType(.float32)).asType(x.dtype)
    }
    static let normalizeQK = compile { (inputs: [MLXArray]) -> [MLXArray] in
        let q = inputs[0], k = inputs[1]
        return [q * rsqrt(sum(square(q), axis: -1, keepDims: true) + 1e-6) / Float(128).squareRoot(),
            k * rsqrt(sum(square(k), axis: -1, keepDims: true) + 1e-6)]
    }
    static let activate = compile { (x: MLXArray) in silu(x / 4) }
    static let injection = compile { (x: MLXArray) in 2 * sigmoid(x / 4) }
    static let swiglu = compile { (gate: MLXArray, up: MLXArray) in silu(gate) * up }
    static let sigmoidGate = compile { (value: MLXArray, gate: MLXArray) in
        value * sigmoid(gate)
    }
    static let floatSigmoidGate = compile { (value: MLXArray, gate: MLXArray) in
        (value.asType(.float32) * sigmoid(gate.asType(.float32))).asType(value.dtype)
    }
    static let mix = compile { (projected: MLXArray, normalized: MLXArray) in
        let shape = [1, projected.dim(1), 4, 2560]
        return mean(sigmoid(projected).reshaped(shape) * normalized.reshaped(shape), axis: 2)
    }
    static let inject = compile { (inputs: [MLXArray]) -> [MLXArray] in
        let x = inputs[0], branch = inputs[1], gates = inputs[2]
        return [x + (branch.reshaped([1, x.dim(1), 1, 2560])
            * gates.reshaped([1, x.dim(1), 4, 1])).reshaped(x.shape)]
    }
}
