#!/usr/bin/env python3
"""Convert official RealESRGAN_x4plus.pth to FP32 OHWI safetensors for Synomancer.
Requires torch and numpy only; used once during setup, never by the App.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import tempfile

import torch


def convert(source: Path, destination: Path):
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    weights = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    expected = {"conv_first": (64, 3), "conv_body": (64, 64),
                "conv_up1": (64, 64), "conv_up2": (64, 64),
                "conv_hr": (64, 64), "conv_last": (3, 64)}
    for block in range(23):
        for dense in range(1, 4):
            for layer in range(1, 6):
                expected[f"body.{block}.rdb{dense}.conv{layer}"] = (
                    64 if layer == 5 else 32, 64 + (layer - 1) * 32)
    if set(weights) != {name + suffix for name in expected for suffix in (".weight", ".bias")}:
        raise ValueError("Expected the unmodified 23-block RealESRGAN_x4plus checkpoint")
    tensors = {}
    for name, (out_channels, in_channels) in expected.items():
        weight, bias = weights[name + ".weight"], weights[name + ".bias"]
        if tuple(weight.shape) != (out_channels, in_channels, 3, 3) or tuple(bias.shape) != (out_channels,):
            raise ValueError(f"Unexpected tensor shape: {name}")
        tensors[name + ".weight"] = weight.permute(0, 2, 3, 1).contiguous().float().numpy().astype("<f4")
        tensors[name + ".bias"] = bias.contiguous().float().numpy().astype("<f4")
    header = {"__metadata__": {"architecture": "RealESRGAN_x4plus", "layout": "OHWI"}}
    offset = 0
    for name, array in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for array in tensors.values():
                output.write(array.tobytes())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Converted {len(tensors)} tensors: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
