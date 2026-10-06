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
]
with tempfile.TemporaryDirectory(prefix='synomancer-swiftdata-') as scratch:
    executable = Path(scratch) / 'swiftdata-tests'
    subprocess.run([
        'swiftc', '-swift-version', '6', '-parse-as-library',
        '-target', 'arm64-apple-macos15.0', '-I', str(products),
        *map(str, sources), str(products / 'Synogging.o'), str(products / 'Logging.o'),
        '-o', str(executable),
    ], check=True, cwd=root)
    subprocess.run([str(executable)], check=True, cwd=root)
