#!/usr/bin/env python3
"""Convert a complete local Diffusers FLUX.2-dev snapshot to native MLX affine Q8.

Requires mlx. No network access. Writes an independent directory atomically;
never changes source weights. VAE and non-matrix tensors keep original precision.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time


def read_json(path):
    return json.loads(path.read_text())


def source_shards(directory):
    indexes = sorted(directory.glob('*.safetensors.index.json'))
    if len(indexes) != 1:
        raise ValueError(f'Expected one safetensors index: {directory}')
    names = sorted(set(read_json(indexes[0])['weight_map'].values()))
    if not names or any(Path(n).name != n or not n.endswith('.safetensors') for n in names):
        raise ValueError('Invalid shard paths')
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return indexes[0].name, names


def convert(source, output, group_size=64):
    import mlx.core as mx

    source, output = source.resolve(), output.absolute()
    index = read_json(source / 'model_index.json')
    if index.get('_class_name') != 'Flux2Pipeline':
        raise ValueError('Source must be an unquantized FLUX.2-dev Diffusers snapshot')
    transformer = read_json(source / 'transformer/config.json')
    encoder = read_json(source / 'text_encoder/config.json')
    if transformer.get('joint_attention_dim') != 15360 or encoder.get('model_type') != 'mistral3':
        raise ValueError('Expected dev transformer and Mistral3 text encoder')
    for config in [transformer, encoder]:
        if 'quantization' in config or 'quantization_config' in config:
            raise ValueError('Source is already quantized; refusing to requantize')
    parts = {part: source_shards(source / part) for part in ['transformer', 'text_encoder']}
    staging = (output.parent.parent / ('.' + output.name + '.partial')
               if output.parent.name == 'snapshots' else output.with_name(output.name + '.partial'))
    if output.exists() or staging.exists():
        raise FileExistsError(f'Refusing to overwrite {output} or {staging}')
    staging.mkdir(parents=True)
    report = {'source': str(source), 'mlx_version': mx.__version__, 'bits': 8,
              'group_size': group_size, 'mode': 'affine', 'components': {}}
    started = time.monotonic()
    for part, (index_name, shard_names) in parts.items():
        src, dst = source / part, staging / part
        dst.mkdir()
        config = read_json(src / 'config.json')
        config['quantization'] = {'bits': 8, 'group_size': group_size, 'mode': 'affine'}
        config['mlx_format'] = True
        (dst / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        weight_map, size, count, max_error = {}, 0, 0, 0.0
        for shard_name in shard_names:
            print(f'{part}/{shard_name}: loading', flush=True)
            original = mx.load(str(src / shard_name))
            converted = {}
            for name in sorted(original):
                value = original[name]
                if name.endswith('.weight') and value.ndim == 2:
                    if value.shape[-1] % group_size:
                        raise ValueError(f'{name}: width not divisible by group size')
                    q, scales, biases = mx.quantize(value, group_size=group_size, bits=8, mode='affine')
                    mx.eval(q, scales, biases)
                    if not bool(mx.all(mx.isfinite(scales)) & mx.all(mx.isfinite(biases))):
                        raise ValueError(f'{name}: nonfinite quantization parameters')
                    # Check a bounded row sample; do not expand whole packed matrices.
                    sample = value[:32].astype(mx.float32)
                    restored = mx.dequantize(q[:32], scales[:32], biases[:32],
                                            group_size=group_size, bits=8).astype(mx.float32)
                    error = float(mx.sqrt(mx.mean((sample - restored) ** 2)) /
                                  mx.maximum(mx.sqrt(mx.mean(sample ** 2)), mx.array(1e-12)))
                    if error > .05:
                        raise ValueError(f'{name}: unexpected relative RMS error {error}')
                    max_error = max(max_error, error)
                    base = name[:-len('.weight')]
                    converted.update({name: q, base + '.scales': scales, base + '.biases': biases})
                    count += 1
                    del sample, restored, q, scales, biases
                else:
                    converted[name] = value
                del value
            mx.save_safetensors(str(dst / shard_name), converted, metadata={'format': 'mlx'})
            weight_map.update({name: shard_name for name in converted})
            size += sum(v.nbytes for v in converted.values())
            print(f'{part}/{shard_name}: saved, cumulative {size / 1e9:.2f} GB', flush=True)
            del original, converted
            gc.collect()
            mx.clear_cache()
        (dst / index_name).write_text(json.dumps({'metadata': {'total_size': size},
                                               'weight_map': weight_map}, indent=2) + '\n')
        for f in src.glob('*.json'):
            if f.name != 'config.json' and not f.name.endswith('.index.json'):
                shutil.copyfile(f, dst / f.name)
        report['components'][part] = {'quantized_matrices': count, 'tensor_bytes': size,
                                     'max_sample_relative_rms_error': max_error}
    for part in ['vae', 'tokenizer', 'scheduler']:
        shutil.copytree(source / part, staging / part, symlinks=False)
    for f in source.iterdir():
        if f.is_file() and f.suffix.lower() in ['.md', '.txt']:
            shutil.copyfile(f, staging / f.name)
    report['seconds'] = time.monotonic() - started
    (staging / 'quantization_report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Publish model_index last; partial conversion is never discoverable.
    (staging / 'model_index.json').write_text(json.dumps(index, indent=2) + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, output)
    print(json.dumps(report, indent=2), flush=True)
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group-size', type=int, choices=[32, 64, 128], default=64)
    args = parser.parse_args()
    convert(args.source, args.output, args.group_size)
