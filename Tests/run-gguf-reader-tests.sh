#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
mkdir -p .build/GGUFTests
common=(Applications/Synomancer/Services/GGUF/GGUFFile.swift Applications/Synomancer/Services/GGUF/Qwen4GGUFDescriptor.swift Applications/Synomancer/Services/LocalChatModelCache.swift)
xcrun swiftc -parse-as-library -module-cache-path .build/TestModuleCache $common Tests/GGUFReaderTests.swift -o .build/GGUFTests/reader
.build/GGUFTests/reader
xcrun swiftc -parse-as-library -module-cache-path .build/TestModuleCache $common Tests/LocalChatCacheTests.swift -o .build/GGUFTests/cache
.build/GGUFTests/cache
