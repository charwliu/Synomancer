import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

/// App-local loading bridge for Qwen3.5 MoE checkpoints that pack an MTP head
/// into the main shards. Upstream 3.32.3 bypasses the usual MTP filtering when
/// safetensor metadata says format=mlx. Preserve that conversion convention
/// (including already shifted norms), then exclude only the draft namespace.
final class LocalQwenMoEModel: Module, MLXLMCommon.LanguageModel {
    private let model: any MLXLMCommon.LanguageModel

    init(_ model: any MLXLMCommon.LanguageModel) {
        self.model = model
        super.init()
    }

    static let factory = VLMModelFactory(
        typeRegistry: ModelTypeRegistry<MLXLMCommon.LanguageModel>(creators: [
            "qwen3_5_moe": { data in
                let config = try JSONDecoder.json5().decode(MLXVLM.Qwen35Configuration.self, from: data)
                return LocalQwenMoEModel(MLXVLM.Qwen35MoE(config))
            }
        ]),
        processorRegistry: VLMProcessorTypeRegistry.shared,
        modelRegistry: VLMRegistry.shared)

    static func mainWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.split(separator: ".").contains("mtp") }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Self.mainWeights(model.sanitize(weights: weights))
    }

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        Self.mainWeights(model.sanitize(weights: weights, metadata: metadata))
    }

    // Delegate the actual module tree, keeping the checkpoint's original paths
    // and strict parameter verification through mixed per-layer quantization.
    override func items() -> ModuleItems { model.items() }

    override func update(parameters: ModuleParameters, verify: VerifyUpdate,
                         path: [String] = [], modulePath: [String] = []) throws -> Self {
        try model.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        return self
    }

    override func update(modules: ModuleChildren, verify: VerifyUpdate,
                         path: [String] = [], modulePath: [String] = []) throws -> Self {
        try model.update(modules: modules, verify: verify, path: path, modulePath: modulePath)
        return self
    }

    override func didSetTrain(_ mode: Bool) { model.train(mode) }
    var toolCallFormat: ToolCallFormat? { model.toolCallFormat }
    var reasoningConfig: ReasoningConfig? { model.reasoningConfig }
    func prepare() throws { try model.prepare() }
    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        try model.prepare(input, cache: cache, state: state, prefill: prefill)
    }
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        model(input, cache: cache, state: state)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { model(inputs, cache: cache) }
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { try model.newCache(parameters: parameters) }
    func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus { try model.cacheStatus(parameters: parameters) }
}
