// Synthetic regression adapted from Apple's MixedPrecisionQuantLoadTests (Apache-2.0).
import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLLM
import MLXLMCommon

@main
struct MLXModelLibraryTests {
    static func main() async throws {
        setbuf(stdout, nil)
        MLXRandom.seed(42)
        for type in ["qwen3", "qwen3_next", "qwen3_5", "qwen3_5_text"] {
            let supported = await LLMTypeRegistry.shared.contains(type)
            precondition(supported, "Missing upgraded architecture: \(type)")
        }
        print("PASS: official registry supports Qwen3, Next and Qwen3.5")
        let config = try JSONDecoder().decode(Qwen2Configuration.self, from: Data("""
            {"hidden_size":64,"num_hidden_layers":1,"intermediate_size":128,
             "num_attention_heads":2,"num_key_value_heads":2,"rms_norm_eps":0.00001,
             "vocab_size":64,"tie_word_embeddings":false}
            """.utf8))
        let reference = Qwen2Model(config)
        var overrides: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
        var eightBit: [String] = [], fourBit: [String] = []
        quantize(model: reference) { path, module in
            guard module is Linear || module is Embedding else { return nil }
            let bits = path.contains(".mlp.") ? 8 : 4
            if bits == 8 { eightBit.append(path); overrides[path] = ["group_size": 64, "bits": 8] }
            else { fourBit.append(path) }
            return (groupSize: 64, bits: bits, mode: .affine)
        }
        precondition(!eightBit.isEmpty && !fourBit.isEmpty)
        let arrays = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("synomancer-mlx-library-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try save(arrays: arrays, url: root.appendingPathComponent("model-00001-of-00001.safetensors"))
        let weightMap = Dictionary(uniqueKeysWithValues: arrays.keys.map { ($0, "model-00001-of-00001.safetensors") })
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap]).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        // A sidecar must not get recursively merged into the main model.
        try FileManager.default.createDirectory(at: root.appendingPathComponent("optiq"), withIntermediateDirectories: true)
        try save(arrays: ["unrelated.mtp.weight": MLXArray.zeros([64, 64])], url: root.appendingPathComponent("optiq/mtp.safetensors"))
        let baseData = try JSONSerialization.data(withJSONObject: ["model_type": "qwen2", "quantization": overrides])
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: baseData)
        let loaded = Qwen2Model(config)
        try await loadWeights(modelDirectory: root, model: loaded, perLayerQuantization: base.perLayerQuantization)
        let modules = Dictionary(uniqueKeysWithValues: loaded.leafModules().flattened())
        for path in eightBit { precondition((modules[path] as? Quantized)?.bits == 8) }
        for path in fourBit { precondition((modules[path] as? Quantized)?.bits == 4) }
        let input = MLXArray([1, 2, 3])[.newAxis]
        let expected = reference(input, cache: nil)
        let actual = loaded(input, cache: nil)
        let difference = abs(actual - expected).max().item(Float.self)
        precondition(difference < 0.00001, "Quantized checkpoint changed logits: \(difference)")
        print("PASS: actual MLX 4/8-bit checkpoint round trip matches logits and excludes nested MTP weights")
        // Reproduce converted format=mlx shards with an inline draft head.
        // Strict loading must still reject unknown *main* parameters; only the
        // exact MTP namespace is excluded by the App bridge.
        let packed = root.appendingPathComponent("packed-mtp")
        try FileManager.default.createDirectory(at: packed, withIntermediateDirectories: true)
        var packedArrays = arrays
        packedArrays["language_model.mtp.layers.0.input_layernorm.weight"] = MLXArray.ones([64])
        packedArrays["mtp.fc.weight"] = MLXArray.zeros([64, 128])
        try save(arrays: packedArrays, metadata: ["format": "mlx"], url: packed.appendingPathComponent("model.safetensors"))
        do {
            try await loadWeights(modelDirectory: packed, model: Qwen2Model(config), perLayerQuantization: base.perLayerQuantization)
            preconditionFailure("Inline MTP unexpectedly loaded into the main model")
        } catch { print("PASS: reproduced strict main-model failure for inline MTP parameters") }
        let wrapped = LocalQwenMoEModel(Qwen2Model(config))
        try await loadWeights(modelDirectory: packed, model: wrapped, perLayerQuantization: base.perLayerQuantization)
        precondition(abs(wrapped(input, cache: nil) - expected).max().item(Float.self) < 0.00001)
        let main = wrapped.parameters().flattened()
        precondition(Set(main.map(\.0)) == Set(arrays.keys), "Wrapper changed checkpoint parameter paths")
        for (key, value) in main where key.contains("norm.weight") {
            precondition(abs(value - arrays[key]!).max().item(Float.self) == 0, "Converted norms were shifted again")
        }
        let retained = LocalQwenMoEModel.mainWeights(["model.attempt.weight": MLXArray.ones([1]), "model.mtp.weight": MLXArray.ones([1])])
        precondition(retained.count == 1 && retained["model.attempt.weight"] != nil)
        print("PASS: inline MTP exclusion preserves converted norms, exact 4/8-bit logits and the original module tree")
        do {
            try await loadWeights(modelDirectory: root, model: Qwen2Model(config), quantization: .init(groupSize: 64, bits: 4))
            preconditionFailure("Ignoring per-layer overrides must fail")
        } catch { print("PASS: uniform 4-bit loading rejects incompatible 8-bit tensors") }
        try LocalQwenMTP.validateGating(["model_type": "qwen3_5", "text_config": ["output_gate_type": "swish"]])
        try LocalQwenMTP.validateGating(["model_type": "qwen3_5_text", "output_gate_type": "silu"])
        do {
            try LocalQwenMTP.validateGating(["model_type": "qwen3_5", "text_config": ["output_gate_type": "unknown"]])
            preconditionFailure("Unknown GDN gate was accepted")
        } catch LocalQwenMTP.AdapterError.unsupportedGate { }
        try testSwishGDN()
        try testSigmoidAttention()
        try await testMTPCheckpoint(root: root)
    }

    // An independent one-token GDN calculation. Initial recurrent state is zero,
    // so the delta rule reduces to beta * v * dot(k, q); no kernel is reused here.
    static func testSwishGDN() throws {
        let data = Data(tinyConfig.utf8)
        let cfg = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data)
        let model = Qwen35TextModel(cfg)
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, MLXArray.zeros($0.1.shape)) })
        let prefix = "model.layers.0.linear_attn."
        for (key, value) in weights where key.contains("norm.weight") { weights[key] = MLXArray.ones(value.shape) }
        weights["model.embed_tokens.weight"] = MLXArray.ones([64, 64])
        weights["lm_head.weight"] = eye(64)
        let z = MLXArray((0..<64).map { Float($0 % 5) - 2 })
        let v = MLXArray((0..<64).map { Float($0 + 1) / 64 })
        weights[prefix + "in_proj_z.weight"] = diag(z)
        weights[prefix + "in_proj_qkv.weight"] = concatenated([MLXArray.ones([128, 64]) / 64, diag(v)], axis: 0)
        weights[prefix + "conv1d.weight"] = concatenated([MLXArray.zeros([192, 3, 1]), MLXArray.ones([192, 1, 1])], axis: 1)
        weights[prefix + "out_proj.weight"] = eye(64)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        let eps: Float = 1e-6
        let u: Float = 1 / sqrt(1 + eps)
        let c = u / (1 + exp(-u))
        let qk: Float = c * c / (c * c + eps) / 8
        let raw = silu(v * u) * (0.5 * qk)
        let normalized = raw / sqrt(mean(raw * raw) + eps)
        let h = 1 + normalized * silu(z * u)
        let expected = h / sqrt(mean(h * h) + eps)
        let actual = model(MLXArray([1])[.newAxis], cache: nil)[0, 0]
        let delta = abs(actual - expected).max().item(Float.self)
        print("SWISH reference max logit delta=\(delta)")
        precondition(delta < 0.00003, "GDN swish logits differ from independent delta-rule reference: \(delta)")
        let wrong = 1 + normalized * sigmoid(z * u)
        let sigmoidLogits = wrong / sqrt(mean(wrong * wrong) + eps)
        precondition(abs(actual - sigmoidLogits).max().item(Float.self) > 0.1)
        print("PASS: upstream text GDN uses swish/SiLU, matches independent logits, and differs from sigmoid")
    }

    static func testSigmoidAttention() throws {
        let cfg = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(tinyConfig.utf8))
        let model = Qwen35TextModel(cfg)
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, MLXArray.zeros($0.1.shape)) })
        for (key, value) in weights where key.contains("norm.weight") { weights[key] = MLXArray.ones(value.shape) }
        weights["model.embed_tokens.weight"] = MLXArray.ones([64, 64])
        weights["lm_head.weight"] = eye(64)
        let prefix = "model.layers.1.self_attn."
        let z = MLXArray((0..<64).map { Float($0 % 5) - 2 })
        let gateRows = diag(z)
        weights[prefix + "q_proj.weight"] = concatenated([
            MLXArray.zeros([32, 64]), gateRows[0..<32],
            MLXArray.zeros([32, 64]), gateRows[32..<64]], axis: 0)
        weights[prefix + "v_proj.weight"] = MLXArray.ones([32, 64]) / 64
        weights[prefix + "o_proj.weight"] = eye(64)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        let u: Float = 1 / sqrt(1.000001)
        let h = 1 + u * sigmoid(z * u)
        let expected = h / sqrt(mean(h * h) + Float(1e-6))
        let actual = model(MLXArray([1])[.newAxis], cache: nil)[0, 0]
        let delta = abs(actual - expected).max().item(Float.self)
        precondition(delta < 0.00003, "Full-attention sigmoid logits mismatch: \(delta)")
        let wrong = 1 + u * silu(z * u)
        precondition(abs(actual - wrong / sqrt(mean(wrong * wrong) + Float(1e-6))).max().item(Float.self) > 0.1)
        print("PASS: full attention retains sigmoid and matches independent single-token SDPA logits")
    }

    static let tinyConfig = """
        {"model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":2,
         "intermediate_size":64,"num_attention_heads":2,"num_key_value_heads":1,
         "head_dim":32,"linear_num_value_heads":1,"linear_num_key_heads":1,
         "linear_key_head_dim":64,"linear_value_head_dim":64,"linear_conv_kernel_dim":4,
         "vocab_size":64,"full_attention_interval":2,"rms_norm_eps":0.000001,
         "tie_word_embeddings":false,"mtp_num_hidden_layers":1,"partial_rotary_factor":1.0}
        """

    static func testMTPCheckpoint(root: URL) async throws {
        let cfg = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(tinyConfig.utf8))
        let reference = Qwen35MTPDraftModel(cfg, preconvertedNorms: true)
        // A raw gamma above 0.5 must also shift; magnitude heuristics are incorrect.
        try reference.update(parameters: ModuleParameters.unflattened(["mtp.norm.weight": MLXArray.full([64], values: MLXArray(Float(2.25)))]), verify: [])
        quantize(model: reference) { path, module in
            guard module is Linear, path != "mtp.fc" else { return nil }
            return (groupSize: 64, bits: 4, mode: .affine)
        }
        let weights = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        var rawWeights = weights
        for (key, weight) in weights where key.hasSuffix("norm.weight") || key.hasSuffix("norm_hidden.weight") || key.hasSuffix("norm_embedding.weight") {
            rawWeights[key] = weight - 1
        }
        try save(arrays: rawWeights, url: root.appendingPathComponent("optiq/mtp.safetensors"))
        var config = try JSONSerialization.jsonObject(with: Data(tinyConfig.utf8)) as! [String: Any]
        config["mtp_file"] = "optiq/mtp.safetensors"
        config["mtp_policy"] = "optiq-int4-prequantized-gs64"
        config["mtplx_mtp_quantization"] = ["prequantized": true, "bits": 4, "group_size": 64, "mode": "affine"] as [String: Any]
        try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("config.json"))
        let checkpoint = try LocalQwenMTP.checkpoint(in: root)!
        let loaded = try LocalQwenMTP.load(checkpoint)
        let parameters = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        for (key, weight) in weights {
            precondition(allClose(parameters[key]!, weight, rtol: 0, atol: 0).item(Bool.self), "MTP tensor changed: \(key)")
        }
        let modules = Dictionary(uniqueKeysWithValues: loaded.leafModules().flattened())
        precondition(modules["mtp.fc"] is Linear && !(modules["mtp.fc"] is Quantized))
        precondition((modules["mtp.layers.0.self_attn.q_proj"] as? Quantized)?.bits == 4)
        print("PASS: OptiQ MTP sidecar converts raw norms once and preserves floating fc, loads packed int4 modules independently")
        let main = Qwen35TextModel(cfg)
        let input = LMInput(text: .init(tokens: MLXArray([1, 2, 3])))
        let params = GenerateParameters(maxTokens: 32, temperature: 0)
        let baseline = try TokenIterator(input: input, model: main, parameters: params)
        let expected = Array(baseline)
        var mtp = try MTPSpeculativeTokenIterator(input: input, mainModel: main, drafter: loaded,
            parameters: params, blockSize: 2)
        var actual: [Int] = []
        while let token = mtp.next() { actual.append(token) }
        precondition(actual == expected, "MTP changed greedy output: \(actual) != \(expected)")
        precondition(mtp.proposedCount > 0 && mtp.passthroughReason == nil)
        print("PASS: actual hybrid GDN/MTP greedy tokens match autoregressive generation with real speculation")
        config.removeValue(forKey: "mtp_policy")
        try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("config.json"))
        do { _ = try LocalQwenMTP.checkpoint(in: root); preconditionFailure("Unknown norm-conversion recipe accepted") }
        catch LocalQwenMTP.AdapterError.unsupportedSidecar { }
        config["mtp_policy"] = "optiq-int4-prequantized-gs64"
        config["mtp_file"] = "../mtp.safetensors"
        try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("config.json"))
        do { _ = try LocalQwenMTP.checkpoint(in: root); preconditionFailure("Sidecar traversal accepted") }
        catch LocalQwenMTP.AdapterError.invalidSidecar { }
        config["mtp_file"] = "optiq/missing.safetensors"
        try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("config.json"))
        let missing = try LocalQwenMTP.checkpoint(in: root)
        precondition(missing == nil)
        print("PASS: missing MTP weights allow fallback and paths outside the checkpoint are rejected")
    }
}
