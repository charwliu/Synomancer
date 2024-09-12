// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-libraries",
    platforms: [.macOS(.v14), .iOS(.v16), .visionOS(.v1)],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "LLM",
            targets: ["MLXLLM"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.16.1"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.11"),
        .package(url: "https://github.com/apple/swift-async-algorithms", from: "1.0.0"),
        .package(url: "https://github.com/SwiftUIX/SwiftUIX.git", from: "0.2.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MLXLLM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
                .product(name: "SwiftUIX", package: "SwiftUIX")
            ],
            path: "Libraries/LLM",
            exclude: [
                "README.md",
                "LLM.h",
            ]
        ),
    ]
)
