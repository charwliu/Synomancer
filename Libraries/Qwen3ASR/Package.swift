// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Qwen3ASR",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [.library(name: "Qwen3ASR", targets: ["Qwen3ASR"]),
               .library(name: "Qwen3TTS", targets: ["Qwen3TTS"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.3.4")),
    ],
    targets: [
        .target(name: "AudioCommon", dependencies: [.product(name: "Hub", package: "swift-transformers")]),
        .target(name: "MLXCommon", dependencies: ["AudioCommon",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift")]),
        .target(name: "Qwen3TTS", dependencies: ["AudioCommon", "MLXCommon",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift")]),
        .target(name: "Qwen3ASR", dependencies: ["AudioCommon", "MLXCommon",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift")]),
    ],
    swiftLanguageModes: [.v5]
)
