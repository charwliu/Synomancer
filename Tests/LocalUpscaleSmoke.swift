import Foundation
import CoreImage
import ImageIO
import MLX

@main
struct LocalUpscaleSmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let fixtures = URL(fileURLWithPath: CommandLine.arguments[2])
        let photo = URL(fileURLWithPath: CommandLine.arguments[3])
        let folder = URL(fileURLWithPath: CommandLine.arguments[4])
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let weights = root.appendingPathComponent(RealESRGANUpscaler.relativePath)
        let model = try RealESRGANUpscaler(url: weights)
        let input = try MLX.loadArray(url: fixtures.appendingPathComponent("input.npy"))
        let expected = try MLX.loadArray(url: fixtures.appendingPathComponent("expected.npy"))
        let actual = try model(input)
        let error = abs(actual - expected).max().item(Float.self)
        precondition(actual.shape == expected.shape && error < 0.001, "MLX differs from PyTorch: \(error)")
        print("PASS RealESRGAN real-weight PyTorch parity, max error \(error)")
        precondition(ImageUpscaleSize.fourK.dimensions(width: 1920, height: 1080) == (3840, 2160))
        precondition(ImageUpscaleSize.fourK.dimensions(width: 1080, height: 1920) == (2160, 3840))
        let service = LocalImageUpscaleService()
        let cache = SpeechCacheAccess.Configuration(path: root.path, bookmark: nil)
        let availability = await service.availability(cache: cache)
        print("Availability Apple supported=\(availability.appleSupported) ready=\(availability.appleReady), RealESRGAN=\(availability.realESRGANReady)")
        precondition(availability.realESRGANReady)
        if CommandLine.arguments.contains("--download-apple") {
            print("Preparing Apple model")
            try await service.prepareAppleModel()
            print("PASS Apple model ready")
        }
        let backend: ImageUpscaleBackend = CommandLine.arguments.contains("--apple") ? .apple : .realESRGAN
        let result = try await service.upscale(Data(contentsOf: photo), backend: backend, size: .fourK, cache: cache) { text, _ in print(text) }
        precondition(max(result.width, result.height) == 3840)
        let decoded = CGImageSourceCreateWithData(result.png as CFData, nil)!
        let image = CGImageSourceCreateImageAtIndex(decoded, 0, nil)!
        precondition(image.width == result.width && image.height == result.height)
        try result.png.write(to: folder.appendingPathComponent(backend.rawValue + "-4k.png"))
        print("PASS \(backend.title) actual PNG \(image.width)x\(image.height), \(result.seconds)s")
        // A transparent, asymmetric input covers alpha restoration and orientation.
        let bounds = CGRect(x: 0, y: 0, width: 80, height: 48)
        let transparent = CIImage(color: .clear).cropped(to: bounds)
        let patch = CIImage(color: CIColor(red: 1, green: 0, blue: 0, alpha: 0.5))
            .cropped(to: CGRect(x: 0, y: 0, width: 40, height: 24)).composited(over: transparent)
        let context = CIContext()
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let data = context.pngRepresentation(of: patch, format: .RGBA8, colorSpace: space)!
        let alphaOutput = try await service.upscale(data, backend: backend, size: .twice, cache: cache) { _, _ in }
        precondition(alphaOutput.width == 160 && alphaOutput.height == 96)
        let alphaImage = CIImage(data: alphaOutput.png)!
        var pixels = [UInt8](repeating: 0, count: 160 * 96 * 4)
        context.render(alphaImage, toBitmap: &pixels, rowBytes: 160 * 4, bounds: alphaImage.extent, format: .RGBA8, colorSpace: space)
        precondition(abs(Int(pixels[(10 * 160 + 10) * 4 + 3]) - 128) <= 2)
        precondition(pixels[(70 * 160 + 120) * 4 + 3] == 0)
        print("PASS 2x sizing, alpha and orientation")
        let task = Task { try await service.upscale(data, backend: backend, size: .fourTimes, cache: cache) { _, _ in } }
        task.cancel()
        do { _ = try await task.value; fatalError("Cancellation ignored") }
        catch is CancellationError { print("PASS cancellation") }
    }
}
