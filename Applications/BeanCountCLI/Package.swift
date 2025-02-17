// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "BeanCountCLI",
    platforms: [
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .executable(
            name: "beancount",
            targets: ["BeanCountCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
        .package(
            url: "https://github.com/scottrhoyt/SwiftyTextTable.git",
            .upToNextMinor(from: "0.9.0")
        ),
        .package(
            url: "https://github.com/onevcat/Rainbow",
            .upToNextMajor(from: "4.0.0")
        ),
        .package(path: "../../Libraries/BeanCountModel"),
        .package(path: "../../Libraries/BeanCountParser"),
        .package(path: "../../Libraries/BeanCountTax"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "BeanCountCLI",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                "BeanCountModel",
                "BeanCountParser",
                "BeanCountTax",
                "SwiftyTextTable",
                "Rainbow"
            ]
        ),
    ]
)
