import Foundation
import MLX

enum Flux2DevTests {
    static func run(_ folder: URL, metadata: URL) async throws {
        let fixture = try MLX.loadArrays(url: folder.appendingPathComponent("fixtures.safetensors"))
        func get(_ name: String) -> MLXArray { fixture[name]! }
        func close(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
            precondition(actual.shape == expected.shape)
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
            precondition(error.isFinite && error < 0.0002, label)
            print("\(label): max absolute error \(error)")
        }
        let tf = folder.appendingPathComponent("transformer")
        let transformer = try Flux2KleinTransformer(weights: QwenImage21Weights(directory: tf),
            config: LocalImageModelCache.json(tf.appendingPathComponent("config.json")))
        for g in [Float(0), Float(4)] {
            close(try transformer(get("latent"), text: get("text"), sigma: .init(0.73), ids: get("ids"), guidance: g),
                get("dit_\(Int(g))"), "FLUX.2-dev embedded guidance \(g)")
        }
        var omitted = try LocalImageModelCache.json(tf.appendingPathComponent("config.json"))
        omitted.removeValue(forKey: "guidance_embeds")
        let defaultGuided = try Flux2KleinTransformer(weights: QwenImage21Weights(directory: tf), config: omitted)
        close(try defaultGuided(get("latent"), text: get("text"), sigma: 0.73, ids: get("ids"), guidance: 4),
            get("dit_4"), "FLUX.2-dev default guidance in original config")
        close(try transformer(concatenated([get("latent"), get("reference")], axis: 1), text: get("text"),
            sigma: 0.73, ids: get("edit_ids"), guidance: 4), get("edit"), "FLUX.2-dev reference attention")
        let textFolder = folder.appendingPathComponent("text_encoder")
        let config = try LocalImageModelCache.json(textFolder.appendingPathComponent("config.json"))
        let original = try QwenImage21Weights(directory: textFolder).arrays
        for prefix in ["language_model.model.", "model.language_model.", "language_model.", "model."] {
            let renamed = Dictionary(uniqueKeysWithValues: original.map {
                (prefix + String($0.key.dropFirst("language_model.model.".count)), $0.value)
            })
            let encoder = try Flux2DevTextEncoder(weights: QwenImage21Weights(arrays: renamed), config: config)
            close(try encoder.encode(get("text_tokens").asArray(Int.self), validCount: 4), get("text_features"),
                "Mistral 10/20/30 hidden states \(prefix)")
            precondition(!encoder.weights.arrays.keys.contains(where: { $0.contains("layers.30.") || $0.contains(".norm.") }))
        }
        let tokens = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("token-tests.json"))) as! [[String: Any]]
        for test in tokens {
            let (actual, valid) = try await Flux2DevTextEncoder.tokens(test["prompt"] as! String,
                directory: metadata.appendingPathComponent("tokenizer"))
            if actual != test["tokens"] as! [Int] || valid != test["validCount"] as! Int {
                let expected = test["tokens"] as! [Int]
                print("TOKEN MISMATCH \((test["prompt"] as! String).debugDescription), count \(valid) vs \(test["validCount"]!)")
                print("actual tail \(actual.suffix(90)); expected tail \(expected.suffix(90))")
            }
            precondition(actual == test["tokens"] as! [Int] && valid == test["validCount"] as! Int,
                         "Official Mistral processor tokenization")
        }
        print("PASS: official Mistral template, padding/truncation, Chinese, emoji and IMG filtering")
        let required = Flux2DevPipeline.requiredMemory(encoderBytes: 48_000_000_000,
            transformerBytes: 64_000_000_000, vaeBytes: 350_000_000)
        precondition(required < 128 * 1024 * 1024 * 1024 / 4 * 3)
        precondition(required > 64 * 1024 * 1024 * 1024 / 4 * 3)
        for request in [LocalImageRequest(modelID: "x", prompt: "x", control: Data([1])),
                        LocalImageRequest(modelID: "x", prompt: "x", guidance: .nan)] {
            do { try Flux2DevPipeline.validate(request); preconditionFailure("Invalid FLUX.2-dev request accepted") }
            catch LocalImageError.invalid { }
        }
        try quantizedParity(folder: folder, fixture: fixture)
        try discovery(folder: folder, metadata: metadata)
        print("PASS: native FLUX.2-dev numerical parity, discovery, memory budget and request checks")
    }

    static func quantizedParity(folder: URL, fixture: [String: MLXArray]) throws {
        func versions(_ arrays: [String: MLXArray]) -> (QwenImage21Weights, QwenImage21Weights) {
            var packed = arrays, expanded = arrays
            for (name, value) in arrays where name.hasSuffix(".weight") && value.ndim == 2 && value.dim(1) % 32 == 0 {
                let q = MLX.quantized(value, groupSize: 32, bits: 8)
                let prefix = String(name.dropLast(".weight".count))
                packed[name] = q.wq; packed[prefix + ".scales"] = q.scales
                packed[prefix + ".biases"] = q.biases!
                expanded[name] = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 32, bits: 8)
            }
            return (QwenImage21Weights(arrays: packed, bits: 8, groupSize: 32), QwenImage21Weights(arrays: expanded))
        }
        func compare(_ a: MLXArray, _ b: MLXArray, _ label: String) {
            let error = abs(a - b).max().item(Float.self)
            print("\(label): packed/dequantized max error \(error)")
            precondition(error.isFinite && error < 0.0002)
        }
        let tf = folder.appendingPathComponent("transformer")
        let config = try LocalImageModelCache.json(tf.appendingPathComponent("config.json"))
        let (packed, expanded) = try versions(QwenImage21Weights(directory: tf).arrays)
        let qa = try Flux2KleinTransformer(weights: packed, config: config)
        let qb = try Flux2KleinTransformer(weights: expanded, config: config)
        let input = concatenated([fixture["latent"]!, fixture["reference"]!], axis: 1)
        compare(try qa(input, text: fixture["text"]!, sigma: 0.73, ids: fixture["edit_ids"]!, guidance: 4),
            try qb(input, text: fixture["text"]!, sigma: 0.73, ids: fixture["edit_ids"]!, guidance: 4), "Q8 dev DiT + reference + guidance")
        let ef = folder.appendingPathComponent("text_encoder")
        let ec = try LocalImageModelCache.json(ef.appendingPathComponent("config.json"))
        let (ep, ed) = try versions(QwenImage21Weights(directory: ef).arrays)
        let a = try Flux2DevTextEncoder(weights: ep, config: ec)
        let b = try Flux2DevTextEncoder(weights: ed, config: ec)
        let tokens = fixture["text_tokens"]!.asArray(Int.self)
        compare(try a.encode(tokens, validCount: 4), try b.encode(tokens, validCount: 4), "Q8 Mistral, packed embedding and left padding")
        let key = a.prefix + "embed_tokens"
        ep.arrays.removeValue(forKey: key + ".biases")
        do {
            _ = try ep.embedding(MLXArray(tokens), key)
            preconditionFailure("Missing quantization biases accepted")
        } catch LocalImageError.unsupported { }
        precondition(LocalImageModelCache.supportsFlux2Quantization([:]))
        precondition(LocalImageModelCache.supportsFlux2Quantization(["quantization": ["bits": 8, "group_size": 64]]))
        for invalid: [String: Any] in [
            ["quantization": ["bits": 4, "group_size": 64]],
            ["quantization": ["bits": 8, "group_size": 7]],
            ["quantization": ["bits": 8, "mode": "mxfp8"]],
            ["quantization_config": ["quant_method": "bitsandbytes"]]
        ] { precondition(!LocalImageModelCache.supportsFlux2Quantization(invalid)) }
        print("PASS: native MLX affine Q8 packed execution and unsupported formats")
    }

    static func discovery(folder: URL, metadata: URL) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("models--black-forest-labs--FLUX.2-dev/snapshots/test")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        for part in ["transformer", "text_encoder", "vae", "tokenizer", "scheduler"] {
            let dst = snapshot.appendingPathComponent(part)
            try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
            for file in try FileManager.default.contentsOfDirectory(at: metadata.appendingPathComponent(part), includingPropertiesForKeys: nil) {
                try FileManager.default.createSymbolicLink(at: dst.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
            }
            if ["transformer", "text_encoder", "vae"].contains(part) {
                try FileManager.default.createSymbolicLink(at: dst.appendingPathComponent("model.safetensors"),
                    withDestinationURL: folder.appendingPathComponent("fixtures.safetensors"))
            }
        }
        try Data(contentsOf: metadata.appendingPathComponent("model_index.json")).write(to: snapshot.appendingPathComponent("model_index.json"))
        let models = try LocalImageModelCache.scan(root)
        precondition(models.count == 1 && models[0].family == .flux2Dev)
        precondition(models[0].family.defaultSteps == 28 && models[0].family.defaultGuidance == 4)
        precondition(models[0].family.supportsReference && !models[0].family.supportsControl)
        for component in ["transformer", "text_encoder"] {
            let file = snapshot.appendingPathComponent(component + "/config.json")
            var config = try LocalImageModelCache.json(file)
            config["quantization"] = ["bits": 8, "group_size": 64, "mode": "affine"]
            try FileManager.default.removeItem(at: file) // replace fixture symlink, never its target
            try JSONSerialization.data(withJSONObject: config).write(to: file)
        }
        let quantized = try LocalImageModelCache.scan(root)
        precondition(quantized.count == 1 && quantized[0].quantized)
        let configURL = snapshot.appendingPathComponent("transformer/config.json")
        var qconfig = try LocalImageModelCache.json(configURL)
        qconfig["quantization"] = ["bits": 4, "group_size": 64]
        try JSONSerialization.data(withJSONObject: qconfig).write(to: configURL)
        let unsupported = try LocalImageModelCache.scan(root)
        precondition(unsupported.isEmpty)
        qconfig["quantization"] = ["bits": 8, "group_size": 64]
        try JSONSerialization.data(withJSONObject: qconfig).write(to: configURL)
        try FileManager.default.removeItem(at: snapshot.appendingPathComponent("text_encoder/model.safetensors"))
        let missing = try LocalImageModelCache.scan(root)
        precondition(missing.isEmpty)
    }
}
