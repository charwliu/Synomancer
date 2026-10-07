#!/usr/bin/env python3
"""Compile native upscale smoke checks using existing Xcode build products."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--derived-data', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--fixtures', type=Path, required=True)
parser.add_argument('--photo', type=Path, required=True)
parser.add_argument('--apple', action='store_true')
parser.add_argument('--download-apple', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
products = args.derived_data.resolve() / 'Build/Products/Debug'
checkouts = root / '.build/SourcePackages/checkouts'
maps = args.derived_data.resolve() / 'Build/Intermediates.noindex/GeneratedModuleMaps'
objects = 'MLX Cmlx MLXFast Numerics RealModule ComplexModule _NumericsShims cLogSupport'.split()
output = products / 'SynomancerUpscaleSmoke'
command = ['swiftc', '-swift-version', '6', '-parse-as-library', '-O',
           '-target', 'arm64-apple-macos26.0', '-module-cache-path', '/private/tmp/SynomancerUpscaleTestModules',
           '-I', str(products), '-I', str(checkouts / 'mlx-swift/Source/Cmlx/include'),
           '-I', str(checkouts / 'mlx-swift/Source/Cmlx/mlx-c'),
           '-I', str(checkouts / 'swift-numerics/Sources/_NumericsShims/include'),
           '-Xcc', '-fmodule-map-file=' + str(maps / 'cLogSupport.modulemap')]
command += [str(root / 'Tests/LocalUpscaleSmoke.swift'), str(root / 'Applications/Synomancer/Services/SpeechCacheAccess.swift')]
base = root / 'Applications/Synomancer/Services/ImageGeneration'
command += [str(base / name) for name in ['RealESRGANUpscaler.swift', 'LocalImageUpscaleService.swift', 'QwenImage21Weights.swift']]
command += [str(products / (name + '.o')) for name in objects]
command += ['-lc++', '-framework', 'Metal', '-framework', 'Accelerate', '-framework', 'CoreImage',
            '-framework', 'ImageIO', '-framework', 'VideoToolbox', '-framework', 'Foundation', '-o', str(output)]
subprocess.run(command, check=True)
subprocess.run([str(output), str(args.cache.resolve()), str(args.fixtures.resolve()), str(args.photo.resolve()),
                str(root / '.build/UpscaleTestOutput'), *(['--apple'] if args.apple else []),
                *(['--download-apple'] if args.download_apple else [])], cwd=products, check=True)
