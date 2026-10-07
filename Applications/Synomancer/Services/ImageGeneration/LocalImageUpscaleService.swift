import Foundation
@preconcurrency import CoreImage
import ImageIO
import UniformTypeIdentifiers
import MLX
#if os(macOS)
@preconcurrency import VideoToolbox
#endif

enum ImageUpscaleBackend: String, CaseIterable, Identifiable, Sendable {
    case apple, realESRGAN
    var id: String { rawValue }
    var title: String { self == .apple ? "Apple VideoToolbox" : "RealESRGAN · MLX" }
}

enum ImageUpscaleSize: String, CaseIterable, Identifiable, Sendable {
    case twice, fourTimes, fourK
    var id: String { rawValue }
    var title: String {
        switch self { case .twice: "2×"; case .fourTimes: "4×"; case .fourK: "4K · 长边 3840" }
    }
    func dimensions(width: Int, height: Int) -> (Int, Int) {
        let scale: Double = self == .fourK ? 3840 / Double(max(width, height)) : self == .twice ? 2 : 4
        return (max(1, Int((Double(width) * scale).rounded())), max(1, Int((Double(height) * scale).rounded())))
    }
}

struct ImageUpscaleResult: Sendable {
    let png: Data
    let width: Int
    let height: Int
    let seconds: Double
    let backend: ImageUpscaleBackend
}

struct ImageUpscaleAvailability: Sendable {
    var appleSupported = false
    var appleReady = false
    var realESRGANReady = false
}

private actor UpscaleModelDownload {
    var outcome: Result<Void, Error>?
    func finish(_ error: Error?) {
        outcome = error.map { .failure($0) } ?? .success(())
    }
}

/// One request at a time; MLX arrays and VT sessions never leave this actor.
actor LocalImageUpscaleService {
    static let shared = LocalImageUpscaleService()
    private var busy = false
    private static let tile = 256
    private static let border = 32

    func availability(cache: SpeechCacheAccess.Configuration) -> ImageUpscaleAvailability {
        var result = ImageUpscaleAvailability()
        if let root = try? cache.resolve() {
            let access = root.startAccessingSecurityScopedResource()
            defer { if access { root.stopAccessingSecurityScopedResource() } }
            result.realESRGANReady = RealESRGANUpscaler.modelURL(root: root) != nil
        }
#if os(macOS)
        if #available(macOS 26, *), let config = try? Self.appleConfiguration() {
            result.appleSupported = true
            result.appleReady = config.configurationModelStatus == .ready
        }
#endif
        return result
    }

    /// Called only by the explicit model download button in the UI.
    func prepareAppleModel() async throws {
        while busy { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(100)) }
        try Task.checkCancellation()
        busy = true; defer { busy = false }
#if os(macOS)
        if #available(macOS 26, *) {
            let config = try Self.appleConfiguration()
            if config.configurationModelStatus == .downloadRequired {
                let download = UpscaleModelDownload()
                Task {
                    do {
                        try await config.downloadConfigurationModel()
                        await download.finish(nil)
                    } catch { await download.finish(error) }
                }
                // Apple owns the background download. Cancellation stops our
                // wait promptly; it does not pretend to cancel the system task.
                while true {
                    try Task.checkCancellation()
                    if let outcome = await download.outcome { try outcome.get(); break }
                    try await Task.sleep(for: .milliseconds(250))
                }
            }
            while config.configurationModelStatus == .downloading {
                try Task.checkCancellation()
                try await Task.sleep(for: .milliseconds(250))
            }
            try Task.checkCancellation()
            guard config.configurationModelStatus == .ready else {
                throw LocalImageError.invalid("Apple 超分模型下载未完成，请重试。")
            }
            return
        }
#endif
        throw LocalImageError.invalid("当前系统或设备不支持 VideoToolbox 图片超分。")
    }

    func upscale(_ data: Data, backend: ImageUpscaleBackend, size: ImageUpscaleSize,
                 cache: SpeechCacheAccess.Configuration,
                 progress: @escaping @Sendable (String, Double) -> Void) async throws -> ImageUpscaleResult {
        while busy { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(100)) }
        try Task.checkCancellation()
        busy = true; defer { busy = false }
        let start = ContinuousClock.now
        guard let decoded = CIImage(data: data, options: [.applyOrientationProperty: true]),
              decoded.extent.width >= 1, decoded.extent.height >= 1,
              decoded.extent.width <= 2048, decoded.extent.height <= 2048 else {
            throw LocalImageError.invalid("超分输入需为不超过 2048×2048 的生成原图。")
        }
        let source = decoded.transformed(by: .init(translationX: -decoded.extent.minX, y: -decoded.extent.minY))
        let width = Int(source.extent.width), height = Int(source.extent.height)
        let (targetWidth, targetHeight) = size.dimensions(width: width, height: height)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [.cacheIntermediates: false, .workingColorSpace: colorSpace])
        // Both models operate on opaque RGB. Restore the original alpha after
        // enhancement, so transparency is resized rather than hallucinated.
        let rgb = source.composited(over: CIImage(color: .white).cropped(to: source.extent))
        var scale = 4
        var realModel: RealESRGANUpscaler?
        var root: URL?
        var access = false
        defer { if access { root?.stopAccessingSecurityScopedResource() } }
#if os(macOS)
        var processor: VTFrameProcessor?
        var appleConfig: VTSuperResolutionScalerConfiguration?
        defer { processor?.endSession() }
#endif
        progress("准备超分模型…", 0)
        switch backend {
        case .realESRGAN:
            let folder = try cache.resolve(); root = folder
            access = folder.startAccessingSecurityScopedResource()
            guard let url = RealESRGANUpscaler.modelURL(root: folder) else {
                throw LocalImageError.invalid("HF_HOME 缺少 \(RealESRGANUpscaler.relativePath)。请先转换本地 RealESRGAN_x4plus 权重，参见 docs/LocalImages.md。")
            }
            realModel = try RealESRGANUpscaler(url: url)
        case .apple:
#if os(macOS)
            if #available(macOS 26, *) {
                let config = try Self.appleConfiguration()
                guard config.configurationModelStatus == .ready else {
                    throw LocalImageError.invalid("请先点击“下载 Apple 超分模型”，完成系统模型准备。")
                }
                let session = VTFrameProcessor()
                try session.startSession(configuration: config)
                processor = session; appleConfig = config; scale = config.scaleFactor
            } else { throw LocalImageError.invalid("Apple 超分需要 macOS 26 或更新版本。") }
#else
            throw LocalImageError.invalid("此版本的 Apple 超分仅支持 macOS。")
#endif
        }
        let outputWidth = width * scale, outputHeight = height * scale
        var output = [UInt8](repeating: 255, count: outputWidth * outputHeight * 4)
        let side = Self.tile + Self.border * 2
        let total = ((width + Self.tile - 1) / Self.tile) * ((height + Self.tile - 1) / Self.tile)
        var completed = 0
        for y in stride(from: 0, to: height, by: Self.tile) {
            for x in stride(from: 0, to: width, by: Self.tile) {
                try Task.checkCancellation()
                let coreWidth = min(Self.tile, width - x), coreHeight = min(Self.tile, height - y)
                let rectangle = CGRect(x: x - Self.border, y: y - Self.border, width: side, height: side)
                let tileImage = rgb.clampedToExtent().cropped(to: rectangle)
                    .transformed(by: .init(translationX: -rectangle.minX, y: -rectangle.minY))
                let enhanced: CIImage
                if let realModel {
                    var floats = [Float](repeating: 0, count: side * side * 4)
                    context.render(tileImage, toBitmap: &floats, rowBytes: side * 16,
                        bounds: CGRect(x: 0, y: 0, width: side, height: side), format: .RGBAf, colorSpace: colorSpace)
                    let input = MLXArray(floats).reshaped(1, side, side, 4)[0..., 0..., 0..., 0..<3]
                    let result = try realModel(input)
                    let alpha = MLXArray.ones([1, side * 4, side * 4, 1])
                    let bytes = MLX.round(clip(concatenated([result, alpha], axis: -1), min: 0, max: 1) * 255).asType(.uint8).asArray(UInt8.self)
                    enhanced = CIImage(bitmapData: Data(bytes), bytesPerRow: side * 4 * 4,
                        size: CGSize(width: side * 4, height: side * 4), format: .RGBA8, colorSpace: colorSpace)
                } else {
#if os(macOS)
                    if #available(macOS 26, *), let processor, let config = appleConfig {
                        enhanced = try await Self.appleTile(tileImage, context: context, colorSpace: colorSpace,
                            processor: processor, configuration: config)
                    } else { throw LocalImageError.invalid("Apple 超分会话不可用。") }
#else
                    throw LocalImageError.invalid("Apple 超分会话不可用。")
#endif
                }
                try Task.checkCancellation()
                let crop = CGRect(x: Self.border * scale, y: Self.border * scale,
                    width: coreWidth * scale, height: coreHeight * scale)
                var tileBytes = [UInt8](repeating: 0, count: coreWidth * scale * coreHeight * scale * 4)
                context.render(enhanced, toBitmap: &tileBytes, rowBytes: coreWidth * scale * 4,
                    bounds: crop, format: .RGBA8, colorSpace: colorSpace)
                let rowBytes = coreWidth * scale * 4
                // Core Image's y is measured from the bottom of the canvas,
                // whereas rendered bitmap rows start at the top. Convert the
                // tile origin only; each tile's rows are already top-to-bottom.
                // Use coreHeight so the partial last tile is placed correctly.
                let destinationTop = (height - y - coreHeight) * scale
                output.withUnsafeMutableBytes { destination in
                    tileBytes.withUnsafeBytes { source in
                        for row in 0..<(coreHeight * scale) {
                            destination.baseAddress!.advanced(by: ((destinationTop + row) * outputWidth + x * scale) * 4)
                                .copyMemory(from: source.baseAddress!.advanced(by: row * rowBytes), byteCount: rowBytes)
                        }
                    }
                }
                completed += 1
                progress("超分处理 \(completed) / \(total)", Double(completed) / Double(total) * 0.9)
            }
        }
        let enhanced = CIImage(bitmapData: Data(output), bytesPerRow: outputWidth * 4,
            size: CGSize(width: outputWidth, height: outputHeight), format: .RGBA8, colorSpace: colorSpace)
        let bounds = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        let resized = Self.resize(enhanced, width: targetWidth, height: targetHeight)
        let alpha = Self.resize(source, width: targetWidth, height: targetHeight)
        let final = resized.applyingFilter("CIBlendWithAlphaMask", parameters: [
            kCIInputBackgroundImageKey: CIImage(color: .clear).cropped(to: bounds), kCIInputMaskImageKey: alpha])
        try Task.checkCancellation()
        progress("编码超分 PNG…", 0.95)
        guard let png = context.pngRepresentation(of: final.cropped(to: bounds), format: .RGBA8,
            colorSpace: colorSpace) else { throw LocalImageError.invalid("无法保存超分 PNG。") }
        try Task.checkCancellation()
        let elapsed = start.duration(to: .now).components
        return .init(png: png, width: targetWidth, height: targetHeight,
            seconds: Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18, backend: backend)
    }

    private static func resize(_ image: CIImage, width: Int, height: Int) -> CIImage {
        let scale = CGFloat(height) / image.extent.height
        return image.applyingFilter("CILanczosScaleTransform", parameters: [
            kCIInputScaleKey: scale, kCIInputAspectRatioKey: CGFloat(width) / image.extent.width / scale])
            .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

#if os(macOS)
    @available(macOS 26, *)
    private static func appleConfiguration() throws -> VTSuperResolutionScalerConfiguration {
        let side = tile + border * 2
        guard VTSuperResolutionScalerConfiguration.isSupported,
              let factor = VTSuperResolutionScalerConfiguration.supportedScaleFactors.filter({ (2...4).contains($0) }).max(),
              let config = VTSuperResolutionScalerConfiguration(frameWidth: side, frameHeight: side,
                scaleFactor: factor, inputType: .image, usePrecomputedFlow: false,
                qualityPrioritization: .normal, revision: VTSuperResolutionScalerConfiguration.defaultRevision) else {
            throw LocalImageError.invalid("此设备不支持 Apple 原生图片超分。可选择 RealESRGAN。")
        }
        return config
    }

    @available(macOS 26, *)
    private static func appleTile(_ image: CIImage, context: CIContext, colorSpace: CGColorSpace,
        processor: VTFrameProcessor, configuration: VTSuperResolutionScalerConfiguration) async throws -> CIImage {
        guard let format = configuration.supportedPixelFormats.first else { throw LocalImageError.unsupported("VideoToolbox pixel format") }
        func buffer(width: Int, height: Int, attributes: [String: any Sendable]) throws -> CVPixelBuffer {
            var result: CVPixelBuffer?
            guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attributes as CFDictionary, &result) == kCVReturnSuccess,
                  let result else { throw LocalImageError.invalid("无法分配 Apple 超分图像缓冲。") }
            return result
        }
        let input = try buffer(width: configuration.frameWidth, height: configuration.frameHeight, attributes: configuration.sourcePixelBufferAttributes)
        let output = try buffer(width: configuration.frameWidth * configuration.scaleFactor,
            height: configuration.frameHeight * configuration.scaleFactor, attributes: configuration.destinationPixelBufferAttributes)
        context.render(image, to: input, bounds: image.extent, colorSpace: colorSpace)
        guard let source = VTFrameProcessorFrame(buffer: input, presentationTimeStamp: .zero),
              let destination = VTFrameProcessorFrame(buffer: output, presentationTimeStamp: .zero),
              let parameters = VTSuperResolutionScalerParameters(sourceFrame: source, previousFrame: nil,
                previousOutputFrame: nil, opticalFlow: nil, submissionMode: .random, destinationFrame: destination) else {
            throw LocalImageError.invalid("无法创建 Apple 超分图像帧。")
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            processor.process(parameters: parameters) { _, error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
        }
        return CIImage(cvPixelBuffer: output, options: [.colorSpace: colorSpace])
    }
#endif
}
