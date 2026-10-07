// Swift/MLX implementation of the RealESRGAN_x4plus RRDBNet architecture.
// Architecture reference: BasicSR, Copyright 2018-2022 BasicSR Authors.
// Apache-2.0; adapted to NHWC tensors and native MLX operations. See docs/LocalImages.md.
import Foundation
import MLX

final class RealESRGANUpscaler {
    static let relativePath = "synomancer/upscalers/RealESRGAN_x4plus.safetensors"
    private let weights: [String: MLXArray]

    static func modelURL(root: URL) -> URL? {
        for base in [root, root.appendingPathComponent("hub")] {
            let url = base.appendingPathComponent(relativePath)
            if FileManager.default.isReadableFile(atPath: url.path) { return url }
        }
        return nil
    }

    init(url: URL) throws {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        guard metadata["architecture"] == "RealESRGAN_x4plus", metadata["layout"] == "OHWI" else {
            throw LocalImageError.invalid("RealESRGAN 权重格式不匹配，请使用 Tools/convert-realesrgan.py 转换官方 x4plus 权重。")
        }
        weights = arrays
        func check(_ name: String, input: Int, output: Int) throws {
            guard arrays[name + ".weight"]?.shape == [output, 3, 3, input],
                  arrays[name + ".bias"]?.shape == [output],
                  arrays[name + ".weight"]?.dtype == .float32,
                  arrays[name + ".bias"]?.dtype == .float32 else {
                throw LocalImageError.missing("RealESRGAN " + name)
            }
        }
        try check("conv_first", input: 3, output: 64)
        for block in 0..<23 {
            for dense in 1...3 {
                for layer in 1...5 {
                    try check("body.\(block).rdb\(dense).conv\(layer)",
                        input: 64 + (layer - 1) * 32, output: layer == 5 ? 64 : 32)
                }
            }
        }
        for name in ["conv_body", "conv_up1", "conv_up2", "conv_hr"] {
            try check(name, input: 64, output: 64)
        }
        try check("conv_last", input: 64, output: 3)
    }

    private func conv(_ x: MLXArray, _ name: String) -> MLXArray {
        MLX.conv2d(x, weights[name + ".weight"]!, padding: 1) + weights[name + ".bias"]!
    }

    private func activate(_ x: MLXArray) -> MLXArray { maximum(x, x * 0.2) }

    func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        guard input.ndim == 4, input.dim(0) == 1, input.dim(3) == 3,
              input.dim(1) > 0, input.dim(2) > 0 else {
            throw LocalImageError.invalid("无效的 RealESRGAN 输入")
        }
        let initial = conv(input.asType(.float32), "conv_first")
        var x = initial
        for block in 0..<23 {
            try Task.checkCancellation()
            let residual = x
            for dense in 1...3 {
                let base = x
                var features = [base]
                for layer in 1...4 {
                    features.append(activate(conv(concatenated(features, axis: -1), "body.\(block).rdb\(dense).conv\(layer)")))
                }
                x = base + conv(concatenated(features, axis: -1), "body.\(block).rdb\(dense).conv5") * 0.2
            }
            x = residual + x * 0.2
            eval(x)
        }
        x = initial + conv(x, "conv_body")
        for name in ["conv_up1", "conv_up2"] {
            x = activate(conv(repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2), name))
            eval(x)
            try Task.checkCancellation()
        }
        let output = conv(activate(conv(x, "conv_hr")), "conv_last")
        eval(output)
        guard all(isFinite(output)).item(Bool.self) else { throw LocalImageError.nonfinite }
        return output
    }
}
