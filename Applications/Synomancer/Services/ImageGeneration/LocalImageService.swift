import Foundation
import CryptoKit
@preconcurrency import CoreImage
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import Tokenizers

struct LocalImageRequest: Sendable {
    var modelID: String
    var prompt: String
    var reference: Data?
    var width = 768
    var height = 768
    var steps = 40
    var seed: UInt64 = 42
    var control: Data?
    var controlScale: Float = 1
    var controlEnd: Float = 1
}

struct LocalImageResult: Sendable {
    let png: Data
    let seed: UInt64
    let seconds: Double
    let modelID: String
}

/// Owns all image arrays on one actor; cancellation joins evaluated GPU work.
/// There is no network downloader, Python process, or server in this service.
actor LocalImageService {
    static let shared = LocalImageService()
    private var busy = false

    // Cache only evaluated conditioning, never encoder weights or denoising KV.
    // Hugging Face snapshot paths identify immutable model revisions.
    private struct ConditioningKey: Equatable {
        let directory: URL
        let prompt: String
        let reference: SHA256.Digest?
        let control: SHA256.Digest?
        let width: Int
        let height: Int
    }
    private struct Conditioning {
        let key: ConditioningKey
        let features: MLXArray
        let slots: [Bool]
        let reference: MLXArray?
        let control: MLXArray?
    }
    private var conditioning: Conditioning?

    func models(cache: SpeechCacheAccess.Configuration) throws -> [LocalImageModel] {
        let root = try cache.resolve()
        let access = root.startAccessingSecurityScopedResource()
        defer { if access { root.stopAccessingSecurityScopedResource() } }
        return try LocalImageModelCache.scan(root)
    }

    func controlAvailability(cache: SpeechCacheAccess.Configuration) throws -> (union: Bool, depth: Bool) {
        let root = try cache.resolve()
        let access = root.startAccessingSecurityScopedResource()
        defer { if access { root.stopAccessingSecurityScopedResource() } }
        return (LocalImageControlCache.find(root) != nil, LocalControlMapCache.depth(root) != nil)
    }

    func generate(_ request: LocalImageRequest, cache: SpeechCacheAccess.Configuration,
                  progress: @escaping @Sendable (String, Double) -> Void) async throws -> LocalImageResult {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              [request.width, request.height].allSatisfy({ $0 >= 256 && $0 <= 2048 && $0 % 32 == 0 }),
              (2...60).contains(request.steps), request.controlScale.isFinite,
              (0...2).contains(request.controlScale), request.controlEnd.isFinite,
              request.controlEnd > 0, request.controlEnd <= 1 else {
            throw LocalImageError.invalid("请输入提示词；图片尺寸需在 256–2048 之间且为 32 的倍数，步数需在 2–60 之间。")
        }
        while busy {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(50))
        }
        try Task.checkCancellation()
        busy = true
        defer { busy = false }
        let root = try cache.resolve()
        let access = root.startAccessingSecurityScopedResource()
        defer { if access { root.stopAccessingSecurityScopedResource() } }
        guard let model = try LocalImageModelCache.scan(root).first(where: { $0.id == request.modelID }) else {
            throw LocalImageError.unavailable
        }
        guard model.bytes < ProcessInfo.processInfo.physicalMemory / 2 else {
            throw LocalImageError.invalid("此图像模型权重过大，请选择 MLX 量化版本。")
        }
        let controlModel: LocalImageControlModel?
        if request.control != nil && request.controlScale > 0 {
            guard let found = LocalImageControlCache.find(root) else {
                throw LocalImageError.invalid("HF_HOME 缺少完整的 alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union 权重。")
            }
            guard model.bytes + found.bytes < ProcessInfo.processInfo.physicalMemory / 2 else {
                throw LocalImageError.invalid("图像模型和 ControlNet 权重合计过大，请使用量化的基础图像模型。")
            }
            controlModel = found
        } else { controlModel = nil }
        let start = ContinuousClock.now
        progress("加载图像模型…", 0)
        let directory = model.directory
        let transformerConfig = try LocalImageModelCache.json(directory.appendingPathComponent("transformer/config.json"))
        guard transformerConfig["in_channels"] as? Int == 64,
              transformerConfig["out_channels"] as? Int == 64,
              transformerConfig["patch_size"] as? Int == 1 else {
            throw LocalImageError.unsupported("64-channel Qwen-Image-2.1")
        }
        let transformer = try QwenImage21Transformer(
            weights: QwenImage21Weights(directory: directory.appendingPathComponent("transformer")),
            config: transformerConfig)
        let control = try controlModel.map { try QwenImage21Control(file: $0.file, transformer: transformer) }
        let vae = try QwenImage21VAE(
            weights: QwenImage21Weights(directory: directory.appendingPathComponent("vae")),
            config: LocalImageModelCache.json(directory.appendingPathComponent("vae/config.json")))
        guard vae.channels == 64 else { throw LocalImageError.unsupported("VAE latent channels") }
        try Task.checkCancellation()
        progress("编码提示词和参考图…", 0.02)
        // Union is structural generation. A reference photo may supply a depth
        // or pose map in the UI; it is not silently fed as ti2i conditioning.
        let referenceData = request.control == nil ? request.reference : nil
        let controlData = control == nil ? nil : request.control
        let key = ConditioningKey(directory: directory.resolvingSymlinksInPath(), prompt: request.prompt,
            reference: referenceData.map { SHA256.hash(data: $0) },
            control: controlData.map { SHA256.hash(data: $0) },
            width: request.width, height: request.height)
        let height = request.height / 16, width = request.width / 16
        let prepared: Conditioning
        if let cached = conditioning, cached.key == key {
            progress("复用提示词和图像编码…", 0.04)
            prepared = cached
        } else {
            conditioning = nil
            let reference = try referenceData.map {
                try Self.referenceImage($0, width: request.width, height: request.height)
            }
            let (features, slots) = try await Self.encodePrompt(request.prompt,
                reference: reference?.image, directory: directory)
            eval(features)
            try Task.checkCancellation()
            var referenceLatent: MLXArray?
            if let reference {
                referenceLatent = try vae.encode(reference.rgba)
                    .reshaped(1, height * width, 64).asType(features.dtype)
                eval(referenceLatent!)
            }
            var controlContext: MLXArray?
            if let data = controlData {
                progress("编码 ControlNet 控制图…", 0.04)
                let pixels = try Self.referenceImage(data, width: request.width,
                    height: request.height, background: .clear)
                let encoded = try vae.encode(pixels.rgba)
                    .reshaped(1, height * width, 64).asType(features.dtype)
                controlContext = concatenated([encoded,
                    MLXArray.zeros([1, height * width, 65], dtype: features.dtype)], axis: -1)
                eval(controlContext!)
            }
            try Task.checkCancellation()
            prepared = Conditioning(key: key, features: features, slots: slots,
                reference: referenceLatent, control: controlContext)
            // One entry, bounded to at most 64 MiB even for FP32 conditioning.
            let elements = features.size + (referenceLatent?.size ?? 0) + (controlContext?.size ?? 0)
            if elements <= 16 * 1024 * 1024 { conditioning = prepared }
        }
        let features = prepared.features
        let dtype = features.dtype
        let referenceLatent = prepared.reference
        let controlContext = prepared.control
        var shapes: [(Int, Int)] = []
        if referenceLatent != nil { shapes.append((height, width)) }
        shapes.append((height, width))
        let layout = try QwenImage21Layout(imageSlots: prepared.slots, shapes: shapes, axes: transformer.axes)
        var latents = MLXRandom.normal([1, height * width, 64], dtype: dtype, key: MLXRandom.key(request.seed))
        let sigmas = try Self.sigmas(steps: request.steps, tokens: height * width,
            config: LocalImageModelCache.json(directory.appendingPathComponent("scheduler/scheduler_config.json")))
        var prefixCache: [(MLXArray, MLXArray)] = []
        var controlCache: [(MLXArray, MLXArray)] = []
        var previousControlEnabled = false
        for step in 0..<request.steps {
            try Task.checkCancellation()
            progress("生成图像 \(step + 1) / \(request.steps)", 0.05 + 0.85 * Double(step) / Double(request.steps))
            let input = referenceLatent.map { concatenated([$0, latents], axis: 1) } ?? latents
            let scale = Float(step) / Float(request.steps) < request.controlEnd ? request.controlScale : 0
            let controlEnabled = control != nil && controlContext != nil && scale > 0
            if controlEnabled != previousControlEnabled {
                // Prefix states depend on whether control hints are injected.
                // Recompute only once when the control schedule crosses its end.
                prefixCache.removeAll()
                controlCache.removeAll()
            }
            let prediction = try transformer.predict(latents: input, text: features,
                timestep: sigmas[step], layout: layout, cache: &prefixCache,
                control: control, controlContext: controlContext, controlScale: scale,
                controlCache: &controlCache)
            previousControlEnabled = controlEnabled
            latents = Self.eulerStep(latents, prediction: prediction,
                delta: sigmas[step + 1] - sigmas[step])
            eval(latents)
        }
        prefixCache.removeAll()
        controlCache.removeAll()
        try Task.checkCancellation()
        progress("解码并保存图像…", 0.92)
        let rgba = try vae.decode(latents.reshaped(1, height, width, 64))
        eval(rgba)
        guard all(isFinite(rgba)).item(Bool.self) else { throw LocalImageError.nonfinite }
        let png = try Self.png(rgba)
        try Task.checkCancellation()
        let duration = start.duration(to: .now).components
        progress("完成", 1)
        return .init(png: png, seed: request.seed, seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18, modelID: model.id)
    }

    /// Accumulate in FP32 so small denoising updates aren't rounded before the
    /// addition. The model's activation stream retains its original dtype.
    static func eulerStep(_ latents: MLXArray, prediction: MLXArray, delta: Float) -> MLXArray {
        (latents.asType(.float32) + prediction.asType(.float32) * MLXArray(delta)).asType(latents.dtype)
    }

    static func sigmas(steps: Int, tokens: Int, config: [String: Any]) throws -> [Float] {
        guard steps >= 2, config["use_dynamic_shifting"] as? Bool == true,
              config["time_shift_type"] as? String == "exponential",
              config["invert_sigmas"] as? Bool != true,
              config["stochastic_sampling"] as? Bool != true else {
            throw LocalImageError.unsupported("FlowMatchEuler scheduler")
        }
        let base = (config["base_image_seq_len"] as? NSNumber)?.doubleValue ?? 256
        let maximum = (config["max_image_seq_len"] as? NSNumber)?.doubleValue ?? 8192
        let baseShift = (config["base_shift"] as? NSNumber)?.doubleValue ?? 0.5
        let maxShift = (config["max_shift"] as? NSNumber)?.doubleValue ?? 0.9
        let mu = baseShift + (Double(tokens) - base) * (maxShift - baseShift) / (maximum - base)
        let factor = exp(mu)
        var values = (0..<steps).map { index -> Double in
            let raw = 1 - Double(index) * (1 - 1 / Double(steps)) / Double(steps - 1)
            return factor / (factor + 1 / raw - 1)
        }
        if let terminal = (config["shift_terminal"] as? NSNumber)?.doubleValue {
            let scale = (1 - values.last!) / (1 - terminal)
            values = values.map { 1 - (1 - $0) / scale }
        }
        return values.map(Float.init) + [0]
    }

    private struct Reference {
        let image: CIImage
        let rgba: MLXArray
    }

    private static func referenceImage(_ data: Data, width: Int, height: Int, background: CIColor = .white) throws -> Reference {
        guard let source = CIImage(data: data, options: [.applyOrientationProperty: true]) else {
            throw LocalImageError.invalid("无法读取参考照片。")
        }
        // Preserve the entire photograph, with an explicit crop in the UI avoided:
        // fit within the selected canvas and letterbox over opaque white.
        let scale = min(CGFloat(width) / source.extent.width, CGFloat(height) / source.extent.height)
        var image = source.transformed(by: .init(translationX: -source.extent.minX, y: -source.extent.minY))
            .transformed(by: .init(scaleX: scale, y: scale))
        image = image.transformed(by: .init(translationX: (CGFloat(width) - image.extent.width) / 2, y: (CGFloat(height) - image.extent.height) / 2))
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        image = image.composited(over: CIImage(color: background).cropped(to: rect)).cropped(to: rect)
        let context = CIContext(options: [.cacheIntermediates: false])
        var pixels = Array(repeating: UInt8(0), count: width * height * 4)
        let color = CGColorSpace(name: CGColorSpace.sRGB)!
        context.render(image, toBitmap: &pixels, rowBytes: width * 4, bounds: rect, format: .RGBA8, colorSpace: color)
        // Pixel data and vision input share this exact resized image.
        let cg = context.createCGImage(image, from: rect)!
        let rgba = MLXArray(pixels).reshaped(1, height, width, 4).asType(.float32) / 127.5 - 1
        return .init(image: CIImage(cgImage: cg), rgba: rgba)
    }

    private static func encodePrompt(_ prompt: String, reference: CIImage?, directory: URL) async throws -> (MLXArray, [Bool]) {
        let folder = directory.appendingPathComponent("text_encoder")
        let config = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: Data(contentsOf: folder.appendingPathComponent("config.json")))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: Data(contentsOf: folder.appendingPathComponent("config.json")))
        let encoder = Qwen3VL(config)
        try await loadWeights(modelDirectory: folder, model: encoder, perLayerQuantization: base.perLayerQuantization)
        // The image checkpoint reads last-layer features BEFORE final RMSNorm.
        // Replace only these two actor-private modules, leaving the official
        // Qwen3-VL vision, M-RoPE and DeepStack implementations intact.
        let dim = config.textConfiguration.hiddenSize
        encoder.update(modules: .unflattened([
            "language_model.model.norm": ImageFeatureNorm(dimensions: dim, eps: 1e-6),
            "language_model.lm_head": ImageFeatureProjection(dim, 1, bias: false)
        ]))
        let processorDirectory = directory.appendingPathComponent("processor")
        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: processorDirectory)
        let system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        let imagePrefix = reference == nil ? "" : "<image1><|vision_start|><|image_pad|><|vision_end|>"
        let raw = system + "<|im_start|>user\n" + imagePrefix + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        let bridge = ImagePromptTokenizer(upstream: tokenizer, raw: raw)
        let processorConfig = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self,
            from: Data(contentsOf: processorDirectory.appendingPathComponent("preprocessor_config.json")))
        let processor = Qwen3VLProcessor(processorConfig, tokenizer: bridge)
        var input = UserInput(prompt: .text(prompt), images: reference.map { [.ciImage($0)] } ?? [])
        if let reference { input.processing.resize = reference.extent.size }
        let prepared = try await processor.prepare(input: input)
        let output = try encoder.prepare(prepared, cache: [], state: nil,
            prefill: .init(stepSize: prepared.text.tokens.size + 1, chunking: .unchunked))
        guard case .logits(let features) = output else { throw LocalImageError.unsupported("Qwen3-VL feature output") }
        let drop = tokenizer.encode(text: system, addSpecialTokens: false).count
        let imageID = tokenizer.convertTokenToId("<|image_pad|>")!
        let ids = prepared.text.tokens.asArray(Int.self)
        let result = features.logits[0..., drop..., 0...]
        eval(result)
        return (result, ids.dropFirst(drop).map { $0 == imageID })
    }

    private static func png(_ rgba: MLXArray) throws -> Data {
        let width = rgba.dim(2), height = rgba.dim(1)
        let pixels = MLX.round(clip((rgba[0] + 1) * 127.5, min: 0, max: 255)).asType(.uint8).asArray(UInt8.self)
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw LocalImageError.invalid("无法编码生成图像。")
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw LocalImageError.invalid("无法创建 PNG。")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw LocalImageError.invalid("无法保存 PNG。") }
        return data as Data
    }
}

private final class ImageFeatureNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

private final class ImageFeatureProjection: Linear {
    override func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

private struct ImagePromptTokenizer: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer
    let raw: String
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { upstream.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] {
        upstream.encode(text: raw, addSpecialTokens: false)
    }
}
