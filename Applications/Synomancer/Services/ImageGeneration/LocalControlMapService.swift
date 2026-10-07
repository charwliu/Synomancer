import Foundation
@preconcurrency import CoreImage
@preconcurrency import CoreML
@preconcurrency import Vision
import ImageIO
import UniformTypeIdentifiers

enum LocalControlMapKind: String, CaseIterable, Identifiable, Sendable {
    case supplied, depth, pose
    var id: String { rawValue }
    var title: String {
        switch self { case .supplied: "直接使用控制图"; case .depth: "自动提取深度"; case .pose: "自动提取人体姿态" }
    }
}

struct LocalControlMapResult: Sendable {
    let png: Data
    let kind: LocalControlMapKind
    let people: Int?
}

enum LocalControlMapCache {
    static let package = "DepthAnythingV2SmallF16.mlpackage"
    static let requiredFiles = ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"]
    static func depth(_ root: URL) -> URL? {
        for hub in [root, root.appendingPathComponent("hub")] {
            let name = "models--apple--coreml-depth-anything-v2-small"
            let repo = hub.lastPathComponent == name ? hub : hub.appendingPathComponent(name)
            var folders = [root]
            if let ref = try? String(contentsOf: repo.appendingPathComponent("refs/main"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
               !ref.isEmpty, !ref.contains("/") { folders.append(repo.appendingPathComponent("snapshots/" + ref)) }
            folders += (try? FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("snapshots"), includingPropertiesForKeys: nil)) ?? []
            for folder in folders {
                let candidate = folder.lastPathComponent == package ? folder : folder.appendingPathComponent(package)
                if requiredFiles.allSatisfy({ name in
                    let file = candidate.appendingPathComponent(name).resolvingSymlinksInPath()
                    return ((try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])).map {
                        $0.isRegularFile == true && ($0.fileSize ?? 0) > 0
                    } ?? false)
                }) { return candidate }
            }
        }
        return nil
    }

    /// Core ML relocates package contents while compiling. HF's relative blob
    /// links would then point outside the relocated package, so copy real files.
    static func materialize(_ package: URL, in directory: URL) throws -> URL {
        let destination = directory.appendingPathComponent(Self.package, isDirectory: true)
        for name in requiredFiles {
            try Task.checkCancellation()
            let source = package.appendingPathComponent(name).resolvingSymlinksInPath()
            let target = destination.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: source, to: target)
        }
        return destination
    }
}

/// Native preprocessing only. Packages are read from HF_HOME; compiled models
/// live in Core ML's temporary directory, never inside the shared HF cache.
actor LocalControlMapService {
    static let shared = LocalControlMapService()
    private var model: MLModel?
    private var modelPath: URL?
    private var compiledURL: URL?
    private var busy = false

    func extract(_ data: Data, kind: LocalControlMapKind, width: Int, height: Int,
                 cache: SpeechCacheAccess.Configuration) async throws -> LocalControlMapResult {
        guard (256...2048).contains(width), (256...2048).contains(height) else { throw LocalImageError.invalid("无效的控制图尺寸") }
        while busy { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(30)) }
        try Task.checkCancellation()
        busy = true; defer { busy = false }
        guard let image = CIImage(data: data, options: [.applyOrientationProperty: true]), !image.extent.isEmpty else {
            throw LocalImageError.invalid("无法读取控制图照片")
        }
        let context = CIContext(options: [.cacheIntermediates: false])
        if kind == .supplied {
            let fitted = Self.fit(image, width: width, height: height)
            guard let cg = context.createCGImage(fitted, from: CGRect(x: 0, y: 0, width: width, height: height),
                format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) else {
                throw LocalImageError.invalid("无法生成控制图")
            }
            return .init(png: try Self.png(cg), kind: kind, people: nil)
        }
        guard let source = context.createCGImage(image, from: image.extent) else { throw LocalImageError.invalid("无法解码控制图照片") }
        if kind == .pose {
            let request = VNDetectHumanBodyPoseRequest()
            try VNImageRequestHandler(cgImage: source, orientation: .up).perform([request])
            try Task.checkCancellation()
            let observations = request.results ?? []
            let rendered = try Self.pose(observations, sourceSize: CGSize(width: source.width, height: source.height), width: width, height: height)
            return .init(png: rendered.png, kind: kind, people: rendered.people)
        }
        let root = try cache.resolve()
        let access = root.startAccessingSecurityScopedResource()
        defer { if access { root.stopAccessingSecurityScopedResource() } }
        guard let package = LocalControlMapCache.depth(root) else {
            throw LocalImageError.invalid("HF_HOME 缺少 apple/coreml-depth-anything-v2-small 的 DepthAnythingV2SmallF16.mlpackage，请先添加完整模型缓存。")
        }
        if model == nil || modelPath != package {
            let staging = FileManager.default.temporaryDirectory.appendingPathComponent("Synomancer-depth-\(UUID())", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: staging) }
            let localPackage = try LocalControlMapCache.materialize(package, in: staging)
            let compiled = try await MLModel.compileModel(at: localPackage)
            do {
                try Task.checkCancellation()
                let configuration = MLModelConfiguration(); configuration.computeUnits = .all
                let loaded = try MLModel(contentsOf: compiled, configuration: configuration)
                let old = compiledURL
                model = loaded; modelPath = package; compiledURL = compiled
                if let old { try? FileManager.default.removeItem(at: old) }
            } catch { try? FileManager.default.removeItem(at: compiled); throw error }
        }
        guard let model, let input = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw LocalImageError.unsupported("Depth Anything Core ML image input")
        }
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                          kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, input.pixelsWide, input.pixelsHigh, input.pixelFormatType, attributes, &buffer) == kCVReturnSuccess,
              let buffer else { throw LocalImageError.invalid("无法创建深度模型输入") }
        let origin = image.transformed(by: .init(translationX: -image.extent.minX, y: -image.extent.minY))
        let resized = origin.transformed(by: .init(scaleX: CGFloat(input.pixelsWide) / origin.extent.width, y: CGFloat(input.pixelsHigh) / origin.extent.height))
        context.render(resized, to: buffer)
        try Task.checkCancellation()
        let features = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
        let result = try await model.prediction(from: features)
        try Task.checkCancellation()
        guard let depth = result.featureValue(for: "depth")?.imageBufferValue else { throw LocalImageError.unsupported("Depth Anything depth output") }
        let map = try Self.depthImage(depth)
        // Restore the source aspect ratio before fitting it to the same canvas
        // as the generation pipeline. Added margins remain transparent.
        let restored = map.transformed(by: .init(scaleX: CGFloat(source.width) / map.extent.width, y: CGFloat(source.height) / map.extent.height))
        let fitted = Self.fit(restored, width: width, height: height)
        guard let cg = context.createCGImage(fitted, from: CGRect(x: 0, y: 0, width: width, height: height),
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) else { throw LocalImageError.invalid("无法生成深度控制图") }
        return .init(png: try Self.png(cg), kind: kind, people: nil)
    }

    static func fit(_ image: CIImage, width: Int, height: Int) -> CIImage {
        let scale = min(CGFloat(width) / image.extent.width, CGFloat(height) / image.extent.height)
        let moved = image.transformed(by: .init(translationX: -image.extent.minX, y: -image.extent.minY))
            .transformed(by: .init(scaleX: scale, y: scale))
            .transformed(by: .init(translationX: (CGFloat(width) - image.extent.width * scale) / 2,
                                  y: (CGFloat(height) - image.extent.height * scale) / 2))
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        return moved.composited(over: CIImage(color: .clear).cropped(to: bounds)).cropped(to: bounds)
    }

    private static func depthImage(_ buffer: CVPixelBuffer) throws -> CIImage {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let address = CVPixelBufferGetBaseAddress(buffer) else { throw LocalImageError.invalid("空的深度输出") }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer), stride = CVPixelBufferGetBytesPerRow(buffer)
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard [kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_OneComponent16Half].contains(format) else { throw LocalImageError.unsupported("Core ML depth pixel format") }
        var values: [Float] = []; values.reserveCapacity(width * height)
        for row in 0..<height {
            let pointer = address.advanced(by: row * stride)
            for column in 0..<width {
                let value = format == kCVPixelFormatType_OneComponent32Float
                    ? pointer.assumingMemoryBound(to: Float.self)[column]
                    : Float(pointer.assumingMemoryBound(to: Float16.self)[column])
                guard value.isFinite else { throw LocalImageError.nonfinite }
                values.append(value)
            }
        }
        guard let low = values.min(), let high = values.max(), high - low > 1e-6 else { throw LocalImageError.invalid("深度模型没有输出有效的空间层次") }
        let pixels = values.map { UInt8(max(0, min(255, (($0 - low) / (high - low) * 255).rounded()))) }
        return CIImage(bitmapData: Data(pixels), bytesPerRow: width, size: CGSize(width: width, height: height),
                       format: .L8, colorSpace: CGColorSpaceCreateDeviceGray())
    }

    private static func pose(_ observations: [VNHumanBodyPoseObservation], sourceSize: CGSize, width: Int, height: Int) throws -> (png: Data, people: Int) {
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw LocalImageError.invalid("无法创建姿态控制图") }
        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
        let names: [VNHumanBodyPoseObservation.JointName] = [.nose, .neck, .rightShoulder, .rightElbow, .rightWrist,
            .leftShoulder, .leftElbow, .leftWrist, .rightHip, .rightKnee, .rightAnkle,
            .leftHip, .leftKnee, .leftAnkle, .rightEye, .leftEye, .rightEar, .leftEar]
        let limbs = [(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]
        let palette: [(CGFloat,CGFloat,CGFloat)] = [(1,0,0),(1,1/3,0),(1,2/3,0),(1,1,0),(2/3,1,0),(1/3,1,0),(0,1,0),(0,1,1/3),(0,1,2/3),
            (0,1,1),(0,2/3,1),(0,1/3,1),(0,0,1),(1/3,0,1),(2/3,0,1),(1,0,1),(1,0,2/3),(1,0,1/3)]
        let scale = min(CGFloat(width) / sourceSize.width, CGFloat(height) / sourceSize.height)
        let size = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        let offset = CGPoint(x: (CGFloat(width) - size.width) / 2, y: (CGFloat(height) - size.height) / 2)
        // Black is part of the pose map's encoding, only within the photo area.
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(origin: offset, size: size))
        var people = 0
        let radius = max(2, CGFloat(min(width, height)) / 128)
        for observation in observations {
            let recognized = try observation.recognizedPoints(.all)
            let points: [CGPoint?] = names.map { name in
                guard let point = recognized[name], point.confidence >= 0.2 else { return nil }
                return CGPoint(x: offset.x + point.location.x * size.width, y: offset.y + (1 - point.location.y) * size.height)
            }
            guard points.compactMap({ $0 }).count >= 5 else { continue }
            people += 1
            context.setLineWidth(radius); context.setLineCap(.round)
            for (index, limb) in limbs.enumerated() {
                guard let a = points[limb.0], let b = points[limb.1] else { continue }
                let color = palette[index]
                context.setStrokeColor(CGColor(red: color.0, green: color.1, blue: color.2, alpha: 0.6))
                context.move(to: a); context.addLine(to: b); context.strokePath()
            }
            for (index, point) in points.enumerated() {
                guard let point else { continue }
                let color = palette[index]
                context.setFillColor(CGColor(red: color.0, green: color.1, blue: color.2, alpha: 1))
                context.fillEllipse(in: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2))
            }
        }
        guard people > 0, let image = context.makeImage() else { throw LocalImageError.invalid("未检测到足够的人体关节点，请选择人物清晰、身体较完整的照片。") }
        return (try png(image), people)
    }

    static func png(_ image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { throw LocalImageError.invalid("无法创建控制图 PNG") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw LocalImageError.invalid("无法保存控制图 PNG") }
        return data as Data
    }
}
