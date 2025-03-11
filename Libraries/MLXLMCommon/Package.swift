// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MLXLMCommon",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .visionOS(.v1),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
    ],
    dependencies: [
        .package(url: "https://github.com/charwliu/mlx-swift", .upToNextMinor(from: "0.21.2")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "0.1.16")),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        
    ]
)
