import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'backport', Path(__file__).resolve().parents[1] / 'Scripts/backport-mlx-metal.py')
backport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backport)


class MetalBackportTests(unittest.TestCase):
    def test_patch_is_idempotent_and_preserves_readonly_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            for relative in backport.HEADERS:
                header = package / relative
                header.parent.mkdir(parents=True, exist_ok=True)
                header.write_text('prefix\n' + backport.OLD + '\nsuffix\n')
                header.chmod(0o444)
            backport.patch_package(package)
            backport.patch_package(package)
            for relative in backport.HEADERS:
                header = package / relative
                self.assertEqual(header.read_text(), 'prefix\n' + backport.NEW + '\nsuffix\n')
                self.assertEqual(header.stat().st_mode & 0o777, 0o444)

    def test_changed_header_rejects_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            for relative, content in zip(backport.HEADERS, [backport.OLD, 'unexpected upstream code']):
                header = package / relative
                header.parent.mkdir(parents=True, exist_ok=True)
                header.write_text(content)
            with self.assertRaises(ValueError):
                backport.patch_package(package)
            self.assertEqual((package / backport.HEADERS[0]).read_text(), backport.OLD)


if __name__ == '__main__':
    unittest.main()
