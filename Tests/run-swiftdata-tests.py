#!/usr/bin/env python3
"""Compile regression checks against the project's existing Xcode build products."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
products = root / '.build/DerivedData/Build/Products/Release'
sources = [
    *sorted((root / 'Applications/Synomancer/Services/DataService').glob('*.swift')),
    *sorted((root / 'Applications/Synomancer/SwiftData/Models').glob('*.swift')),
    root / 'Applications/Synomancer/Extensions/ModelContext+Extension.swift',
    root / 'Applications/Synomancer/Utilities/Logging.swift',
    root / 'Applications/Synomancer/Models/LanguageModel.swift',
    root / 'Tests/SwiftDataServiceTests.swift',
    root / 'Applications/Synomancer/Services/AssistantMessageContent.swift',
]
with tempfile.TemporaryDirectory(prefix='synomancer-swiftdata-') as scratch:
    executable = Path(scratch) / 'swiftdata-tests'
    subprocess.run([
        'swiftc', '-swift-version', '6', '-parse-as-library',
        '-module-name', 'SynomancerProviderRegression',
        '-target', 'arm64-apple-macos15.0', '-I', str(products),
        *map(str, sources), str(products / 'Synogging.o'), str(products / 'Logging.o'),
        '-o', str(executable),
    ], check=True, cwd=root)
    subprocess.run([str(executable)], check=True, cwd=root)
    store = str(Path(scratch) / 'providers.store')
    subprocess.run([str(executable), '--seed-providers', store], check=True, cwd=root)
    for _ in range(2):
        subprocess.run([str(executable), '--verify-providers', store], check=True, cwd=root)

    # Seed the actual prior schema in a separate process, then open it with the
    # new schema. In-memory containers cannot exercise this migration path.
    legacy_test = Path(scratch) / 'LegacyWriter.swift'
    test_source = (root / 'Tests/SwiftDataServiceTests.swift').read_text()
    test_source = '\n'.join(line for line in test_source.splitlines() if 'model.providerID = nil' not in line)
    legacy_test.write_text(test_source.replace('.provider', '.modelProvider').replace('"local-folder"', '"mlx-community/local-folder"'))
    legacy_sources = [root / 'Tests/Fixtures/LegacyLanguageModelSD.swift' if path.name == 'LanguageModelSD.swift'
                      else legacy_test if path.name == 'SwiftDataServiceTests.swift' else path for path in sources]
    legacy_executable = Path(scratch) / 'legacy-writer'
    subprocess.run([
        'swiftc', '-swift-version', '6', '-parse-as-library',
        '-module-name', 'SynomancerProviderRegression',
        '-target', 'arm64-apple-macos15.0', '-I', str(products),
        *map(str, legacy_sources), str(products / 'Synogging.o'), str(products / 'Logging.o'),
        '-o', str(legacy_executable),
    ], check=True, cwd=root)
    legacy_store = str(Path(scratch) / 'legacy.store')
    subprocess.run([str(legacy_executable), '--seed-providers', legacy_store], check=True, cwd=root)
    for _ in range(2):
        subprocess.run([str(executable), '--verify-providers', legacy_store], check=True, cwd=root)
