"""Small real-MLX conversion tests; run with the same Python environment as the tool."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx

spec = importlib.util.spec_from_file_location('quantize_flux2_dev',
    Path(__file__).resolve().parents[1] / 'Tools/quantize-flux2-dev.py')
quantizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantizer)


class Flux2QuantizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.output = self.root / 'models--local--FLUX.2-dev-MLX-8bit/snapshots/test'
        (self.source / 'model_index.json').write_text(json.dumps({'_class_name': 'Flux2Pipeline'}))
        for part, config in [('transformer', {'joint_attention_dim': 15360}),
                             ('text_encoder', {'model_type': 'mistral3'})]:
            folder = self.source / part
            folder.mkdir()
            (folder / 'config.json').write_text(json.dumps(config))
            name = 'model.embed_tokens.weight' if part == 'text_encoder' else 'proj.weight'
            mx.save_safetensors(str(folder / 'model.safetensors'),
                {name: mx.random.normal((64, 64)), 'norm.weight': mx.ones((64,))})
            (folder / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {name: 'model.safetensors', 'norm.weight': 'model.safetensors'}}))
        for part in ['vae', 'tokenizer', 'scheduler']:
            folder = self.source / part
            folder.mkdir()
            (folder / 'config.json').write_text('{}')
        mx.save_safetensors(str(self.source / 'vae/model.safetensors'),
                            {'conv.weight': mx.ones((4, 3, 3, 3), dtype=mx.bfloat16)})

    def test_round_trip_and_no_overwrite(self):
        original = (self.source / 'transformer/model.safetensors').read_bytes()
        vae = (self.source / 'vae/model.safetensors').read_bytes()
        quantizer.convert(self.source, self.output)
        converted = mx.load(str(self.output / 'transformer/model.safetensors'))
        self.assertEqual(converted['proj.weight'].dtype, mx.uint32)
        self.assertEqual(converted['proj.weight'].shape, (64, 16))
        restored = mx.dequantize(converted['proj.weight'], converted['proj.scales'],
                                converted['proj.biases'], group_size=64, bits=8)
        base = mx.load(str(self.source / 'transformer/model.safetensors'))['proj.weight']
        self.assertLess(float(mx.sqrt(mx.mean((restored - base) ** 2))), .02)
        index = json.loads((self.output / 'transformer/model.safetensors.index.json').read_text())
        self.assertEqual(set(index['weight_map']), set(converted))
        self.assertEqual((self.source / 'transformer/model.safetensors').read_bytes(), original)
        self.assertEqual((self.output / 'vae/model.safetensors').read_bytes(), vae)
        config = json.loads((self.output / 'text_encoder/config.json').read_text())
        self.assertEqual(config['quantization'], {'bits': 8, 'group_size': 64, 'mode': 'affine'})
        with self.assertRaises(FileExistsError):
            quantizer.convert(self.source, self.output)

    def test_failed_conversion_is_not_published(self):
        mx.save_safetensors(str(self.source / 'transformer/model.safetensors'),
                            {'proj.weight': mx.ones((64, 63))})
        with self.assertRaisesRegex(ValueError, 'width not divisible'):
            quantizer.convert(self.source, self.output)
        self.assertFalse(self.output.exists())
        staging = self.output.parent.parent / '.test.partial'
        self.assertTrue(staging.exists())
        self.assertFalse((staging / 'model_index.json').exists())


if __name__ == '__main__':
    unittest.main()
