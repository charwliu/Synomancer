# Legacy Synomancer adaptation

The App now uses official `mlx-swift-lm` 3.32.3 via SPM. This legacy package
is not linked into Synomancer; see `../../docs/MLXModelLibrary.md`.

This is the existing vendored Apple MLX language-model common library.
Original copyright and license headers are retained.

`ModelTypeRegistry.supportedModelTypes` exposes the registered architecture keys
under the registry's existing lock. Synomancer uses these keys to discover
supported local Hugging Face snapshots without constructing models or fetching
remote model metadata. Model registration and creation behavior are unchanged.
