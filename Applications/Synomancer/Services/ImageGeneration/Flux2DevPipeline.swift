// FLUX.2-dev uses the shared native Flux2 DiT, RGB VAE and reference packing.
// Diffusers/Transformers Apache-2.0; MFLUX portions MIT. See docs/LocalImages.md.
import Foundation
import MLX

enum Flux2DevPipeline {
    static func requiredMemory(encoderBytes: UInt64, transformerBytes: UInt64, vaeBytes: UInt64) -> UInt64 {
        // Sequential encoder and denoising lifetimes. Reserve space for VAE
        // FP32 upcasting, 2048px activations and Metal allocations.
        max(encoderBytes, transformerBytes + vaeBytes * 2) + 16 * 1024 * 1024 * 1024
    }

    static func validate(_ request: LocalImageRequest) throws {
        try request.validateNegativePrompt(for: .flux2Dev)
        guard request.control == nil else {
            throw LocalImageError.invalid("FLUX.2-dev 不支持 Qwen ControlNet，请清除结构控制图。")
        }
        guard request.guidance.isFinite && (0...10).contains(request.guidance) else {
            throw LocalImageError.invalid("FLUX.2-dev Guidance 需在 0–10 之间。")
        }
    }

    static func checkMemory(_ model: LocalImageModel) throws {
        func bytes(_ component: String) throws -> UInt64 {
            try LocalImageModelCache.shards(model.directory.appendingPathComponent(component)).reduce(0) {
                $0 + UInt64(try $1.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            }
        }
        let required = try requiredMemory(encoderBytes: bytes("text_encoder"),
            transformerBytes: bytes("transformer"), vaeBytes: bytes("vae"))
        let budget = ProcessInfo.processInfo.physicalMemory / 4 * 3
        guard required + UInt64(Memory.activeMemory) < budget else {
            throw LocalImageError.invalid("FLUX.2-dev 推理所需内存超过当前预算，请释放其他已加载的模型或选择 MLX 8bit 权重。")
        }
    }

    static func generate(_ request: LocalImageRequest, model: LocalImageModel,
                         progress: @escaping @Sendable (String, Double) -> Void) async throws -> LocalImageResult {
        try validate(request)
        try checkMemory(model)
        return try await Flux2KleinPipeline.generate(request, model: model, progress: progress)
    }
}
