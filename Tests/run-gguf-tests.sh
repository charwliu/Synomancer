#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
products="$PWD/.build/DerivedData/Build/Products/Release"
checkouts="$PWD/.build/SourcePackages/checkouts"
maps="$PWD/.build/DerivedData/Build/Intermediates.noindex/GeneratedModuleMaps"
objects=(MLX MLXNN MLXFast MLXRandom MLXLMCommon MLXLLM MLXVLM MLXOptimizers Cmlx Numerics ComplexModule RealModule _NumericsShims cLogSupport Logging Tokenizers Hub Jinja OrderedCollections DequeModule InternalCollectionsUtilities yyjson HuggingFace Crypto EventSource)
links=()
for name in $objects; do links+=("$products/$name.o"); done
xcrun swiftc -swift-version 5 -O -whole-module-optimization -parse-as-library \
  -module-cache-path "$PWD/.build/TestModuleCache" -target arm64-apple-macos27.0 \
  -sdk "$(xcrun --show-sdk-path)" -I "$products" \
  -I "$checkouts/mlx-swift/Source/Cmlx/include" -I "$checkouts/mlx-swift/Source/Cmlx/mlx-c" \
  -I "$checkouts/swift-numerics/Sources/_NumericsShims/include" \
  -Xcc "-fmodule-map-file=$maps/cLogSupport.modulemap" -Xcc "-fmodule-map-file=$maps/yyjson.modulemap" \
  Applications/Synomancer/Services/GGUF/*.swift \
  Applications/Synomancer/Services/LocalChatService.swift \
  Applications/Synomancer/Services/LocalMuseGlimmerOutput.swift \
  Applications/Synomancer/Services/LocalQwenDenseOptimization.swift \
  Applications/Synomancer/Services/LocalChatModelCache.swift \
  Applications/Synomancer/Services/LocalChatTokenizerLoader.swift \
  Applications/Synomancer/Services/LocalQwenMoEModel.swift \
  Applications/Synomancer/Services/LocalQwenMTP.swift \
  Applications/Synomancer/Services/AssistantMessageContent.swift \
  Applications/Synomancer/Services/SpeechCacheAccess.swift \
  Tests/Qwen4GGUFSmoke.swift Tests/GGUFKernelTests.swift \
  $links -lc++ -framework Metal -framework Foundation -framework Accelerate -framework CoreImage -framework AVFoundation -framework ImageIO \
  -o "$products/Qwen4GGUFSmoke"
cd "$products"
./Qwen4GGUFSmoke "$@"
