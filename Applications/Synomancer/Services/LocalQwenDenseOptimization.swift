import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

/// Packs compatible dense gate/up projections without dequantizing or changing
/// their scales. Install before the model's first compiled forward.
enum LocalQwenDenseOptimization {
    @discardableResult
    static func install(on model: any MLXLMCommon.LanguageModel) throws -> Int {
        guard model is Qwen35Model || model is Qwen35TextModel else { return 0 }
        var count = 0
        try model.visit { _, parent in
            guard let module = parent.children().flattened().first(where: { $0.0 == "mlp" })?.1 else { return }
            let children = Dictionary(uniqueKeysWithValues: module.children().flattened())
            guard let gate = children["gate_proj"] as? QuantizedLinear,
                  let up = children["up_proj"] as? QuantizedLinear,
                  let down = children["down_proj"] as? Linear,
                  let fused = FusedMLP(gate: gate, up: up, down: down) else { return }
            try parent.update(modules: .unflattened([("mlp", fused as Module)]), verify: .noUnusedKeys)
            count += 1
        }
        return count
    }

    final class FusedMLP: Module, UnaryLayer {
        @ModuleInfo var gateUp: QuantizedLinear
        @ModuleInfo var down: Linear
        let width: Int

        init?(gate: QuantizedLinear, up: QuantizedLinear, down: Linear) {
            guard gate.mode == .affine, up.mode == .affine,
                  gate.bits == up.bits, gate.groupSize == up.groupSize,
                  gate.weight.shape == up.weight.shape,
                  gate.scales.shape == up.scales.shape,
                  gate.bias == nil, up.bias == nil,
                  gate.globalScale == nil, up.globalScale == nil,
                  let gateBiases = gate.biases, let upBiases = up.biases,
                  gateBiases.shape == upBiases.shape else { return nil }
            width = gate.shape.0
            let packed = QuantizedLinear(
                weight: concatenated([gate.weight, up.weight], axis: 0),
                scales: concatenated([gate.scales, up.scales], axis: 0),
                biases: concatenated([gateBiases, upBiases], axis: 0),
                groupSize: gate.groupSize, bits: gate.bits)
            // Materialize once, releasing the concatenation's source references.
            eval(packed)
            _gateUp.wrappedValue = packed
            _down.wrappedValue = down
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let projected = gateUp(x)
            return down(silu(projected[.ellipsis, ..<width]) * projected[.ellipsis, width...])
        }
    }
}
