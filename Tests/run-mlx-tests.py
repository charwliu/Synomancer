#!/usr/bin/env python3
"""Exercise upstream MLX kernels and expert quantization using existing Xcode products."""
from pathlib import Path
import argparse
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--derived-data', type=Path, default=root / '.build/DerivedData')
parser.add_argument('--source-packages', type=Path, default=root / '.build/SourcePackages')
parser.add_argument('--configuration', choices=['Debug', 'Release'], default='Release')
args = parser.parse_args()
products = args.derived_data / 'Build/Products' / args.configuration
mlx = args.source_packages / 'checkouts/mlx-swift/Source'
maps = args.derived_data / 'Build/Intermediates.noindex/GeneratedModuleMaps'
executable = products / 'SynomancerMLXRegression'
objects = ['MLX', 'MLXNN', 'MLXRandom', 'MLXFast', 'Cmlx', 'cLogSupport', 'Numerics', 'ComplexModule', 'RealModule', '_NumericsShims']
subprocess.run([
    'swiftc', '-swift-version', '6', '-parse-as-library', '-target', 'arm64-apple-macos15.0',
    '-I', str(products), '-I', str(mlx / 'Cmlx/include'),
    '-I', str(mlx / 'Cmlx/mlx-c'),
    '-I', str(args.source_packages / 'checkouts/swift-numerics/Sources/_NumericsShims/include'),
    '-Xcc', '-fmodule-map-file=' + str(maps / 'cLogSupport.modulemap'),
    str(root / 'Libraries/MLXLLM/Sources/MLXLLM/SwitchLayers.swift'),
    str(root / 'Tests/MLXCompatibilityTests.swift'),
    *[str(products / (name + '.o')) for name in objects],
    '-lc++', '-framework', 'Accelerate', '-framework', 'Metal', '-framework', 'Foundation',
    '-o', str(executable),
], check=True, cwd=root)
subprocess.run([str(executable)], check=True, cwd=products)
