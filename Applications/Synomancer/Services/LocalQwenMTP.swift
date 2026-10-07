import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

/// OptiQ packs the projections but preserves raw HF zero-centered RMSNorm
/// weights. Unlike the converted target, those norms require one +1 shift.
/// Sidecar quantization is independent of the target's mixed 4/8-bit policy.
enum LocalQwenMTP {
    static let enabledKey = "localChatMTPEnabled"
    static let temperatureKey = "localChatTemperature"

    struct Checkpoint: Sendable {
        let file: URL
        let configuration: Qwen35TextConfiguration
        let quantization: BaseConfiguration.Quantization
        let bytes: UInt64
    }

    static func validateGating(_ config: [String: Any]) throws {
        guard (config["model_type"] as? String)?.hasPrefix("qwen3_5") == true else { return }
        let text = config["text_config"] as? [String: Any] ?? config
        if let gate = text["output_gate_type"] as? String,
           !["swish", "silu"].contains(gate.lowercased()) {
            throw AdapterError.unsupportedGate(gate)
        }
        // Swish applies to GDN RMSNormGated. Full attention retains sigmoid,
        // matching Transformers and the upstream Qwen35 implementation.
    }

    static func checkpoint(in directory: URL) throws -> Checkpoint? {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        try validateGating(config)
        guard ["qwen3_5", "qwen3_5_text"].contains(config["model_type"] as? String ?? ""),
              config["vision_config"] as? [String: Any] == nil else { return nil }
        let text = config["text_config"] as? [String: Any] ?? config
        guard (text["mtp_num_hidden_layers"] as? Int ?? 0) > 0 else { return nil }
        let extras = config["mlx_lm_extra_tensors"] as? [String: Any]
        guard let path = config["mtp_file"] as? String ?? extras?["mtp_file"] as? String else { return nil }
        guard !path.hasPrefix("/"), !path.split(separator: "/").contains(".."),
              path.hasSuffix(".safetensors") else { throw AdapterError.invalidSidecar }
        let file = directory.appendingPathComponent(path)
        // Incomplete HF downloads and dangling blob symlinks leave main inference usable.
        guard FileManager.default.isReadableFile(atPath: file.path) else { return nil }
        // This policy identifies OptiQ's raw-norm sidecar recipe. A generic
        // prequantized head may already have shifted norms and is not interchangeable.
        guard config["mtp_policy"] as? String == "optiq-int4-prequantized-gs64",
              let quantization = config["mtplx_mtp_quantization"] as? [String: Any],
              quantization["prequantized"] as? Bool == true,
              quantization["mode"] as? String == "affine",
              quantization["bits"] as? Int == 4,
              quantization["group_size"] as? Int == 64 else {
            throw AdapterError.unsupportedSidecar
        }
        let textData = try JSONSerialization.data(withJSONObject: text)
        let bytes = try file.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        return Checkpoint(file: file,
            configuration: try JSONDecoder().decode(Qwen35TextConfiguration.self, from: textData),
            quantization: .init(groupSize: 64, bits: 4), bytes: UInt64(max(bytes, 0)))
    }

    static func load(_ checkpoint: Checkpoint) throws -> Qwen35MTPDraftModel {
        let drafter = Qwen35MTPDraftModel(checkpoint.configuration, preconvertedNorms: false)
        let weights = drafter.sanitize(weights: try loadArrays(url: checkpoint.file))
        quantize(model: drafter) { path, _ in
            // In particular, mtp.fc stays BF16; only packed modules have scales.
            weights["\(path).scales"] == nil ? nil : checkpoint.quantization.asTuple
        }
        try drafter.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        drafter.train(false)
        eval(drafter)
        return drafter
    }

    enum AdapterError: LocalizedError {
        case unsupportedGate(String), invalidSidecar, unsupportedSidecar
        var errorDescription: String? {
            switch self {
            case .unsupportedGate(let gate): "Unsupported Qwen GDN output gate: \(gate). This adapter supports swish/SiLU."
            case .invalidSidecar: "The Qwen MTP weight file must be a relative safetensors path inside the model directory."
            case .unsupportedSidecar: "This Qwen MTP sidecar is not an OptiQ 4-bit affine checkpoint with group size 64. Disable MTP to use the main model."
            }
        }
    }
}
