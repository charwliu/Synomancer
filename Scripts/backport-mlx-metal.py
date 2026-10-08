#!/usr/bin/env python3
"""Backport upstream MLX's empty-vector Metal binding fix until Swift picks it up."""
import argparse
import os
from pathlib import Path
import stat

OLD = '''    get_command_encoder()->setBytes(
        vec.data(), nelems * sizeof(typename Vec::value_type), idx);'''
NEW = '''    using T = typename Vec::value_type;
    if (nelems > 0) {
      get_command_encoder()->setBytes(vec.data(), nelems * sizeof(T), idx);
    } else {
      T val{};
      get_command_encoder()->setBytes(&val, sizeof(T), idx);
    }'''
HEADERS = (
    'Source/Cmlx/mlx/mlx/backend/metal/device.h',
    'Source/Cmlx/include-framework/mlx-backend-metal-device.h',
)


def patched_text(text):
    if NEW in text:
        return text
    if text.count(OLD) != 1:
        raise ValueError('MLX Metal header changed; review the upstream backport before building')
    return text.replace(OLD, NEW, 1)


def patch_package(package):
    # Validate both headers before modifying either one.
    changes = []
    for relative in HEADERS:
        header = package / relative
        original = header.read_text()
        updated = patched_text(original)
        if updated != original:
            changes.append((header, updated))
    for header, updated in changes:
        mode = stat.S_IMODE(header.stat().st_mode)
        try:
            header.chmod(mode | stat.S_IWUSR)
            header.write_text(updated)
        finally:
            header.chmod(mode)
        print(f'Applied upstream MLX empty-vector Metal fix: {header}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-packages', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    roots = [project / '.build/SourcePackages']
    if args.source_packages:
        roots.insert(0, args.source_packages)
    if os.environ.get('BUILD_DIR'):
        roots.insert(0, Path(os.environ['BUILD_DIR']).parents[1] / 'SourcePackages')
    packages = {root.resolve() / 'checkouts/mlx-swift' for root in roots}
    found = [package for package in sorted(packages) if package.is_dir()]
    if not found:
        parser.error('No resolved mlx-swift checkout found; resolve package dependencies first')
    for package in found:
        patch_package(package)


if __name__ == '__main__':
    main()
