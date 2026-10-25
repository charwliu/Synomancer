# Synomancer contributor guidelines

## Layout

- `Applications/Synomancer/`: SwiftUI app, actor-owned services, stores and SwiftData models.
- `Libraries/`: local packages and historical adapters. The active MLX chat/VLM implementation
  comes from the SPM revisions pinned by `Synomancer.xcodeproj`, not the old MLX source copies.
- `Tests/`: standalone Swift regression executables, Python build runners and numerical fixtures.
- `Scripts/` and `Tools/`: build patches and offline model preparation tools.
- `docs/`: feature setup, supported formats and validation instructions.
- `.build/`: local dependencies, build products, generated fixtures and test output; never commit it.

## Build and validation

Use the Release command in `README.md`. Keep Xcode derived data in `.build/DerivedData`
and package checkouts in `.build/SourcePackages`. Preserve pinned package versions unless
upgrading them is part of the task. Signing teams and credentials are local configuration.

Choose focused checks from the affected feature documentation. Common commands after building:

```sh
python3 Tests/run-swiftdata-tests.py
Tests/run-gguf-tests.sh --kernels
python3 Tests/run-asr-tests.py --derived-data .build/DerivedData \
  --source-packages .build/SourcePackages --configuration Release --chat-library
```

Run tests that compile or use `SynomancerASRSmoke` sequentially. GPU tests require Metal;
missing hardware, unavailable downloads and skipped cases are limitations, not passes.
Clearly distinguish synthetic fixtures from full-model inference. Inspect generated media
when changing image behavior, and keep actual-model regressions for protocol/numerical changes.

## Implementation

Use four-space Swift indentation and the surrounding naming/style conventions. Keep model and
cache ownership inside the existing actors. Preserve serialization of chat/image GPU work,
cancellation cleanup, token accounting and immediate restart behavior. Do not replace native
Swift/MLX inference with a Python runtime or external inference server.

SwiftData migration and model discovery must preserve existing conversations. Never erase a
database to recover from an open/migration failure. Keep upstream licenses and adaptation notes.

## Public repository hygiene

Before committing, inspect the staged paths and diff; run both `git diff --check` and
`git diff --cached --check`. Stage explicit paths and preserve unrelated local changes.
Use concise commit subjects that cover the affected components.

Do not commit private `.env` files, signing teams/profiles, Keychain exports, API/HF tokens,
SSH keys, personal filesystem paths, model weights, databases, recordings, generated media,
build caches, acceptance logs or raw validation output. Use generic paths and environment
variables in examples. Keep runtime secrets in the established Keychain/in-memory flows.
Small intentional synthetic source fixtures are allowed; downloaded assets remain local.

`.gitignore` does not remove files already tracked by Git. Audit the staged tree and the
history being published for sensitive data and generated artifacts before a push. When a
history rewrite is authorized, create a local backup, rewrite in an isolated repository,
verify the cleaned tree/history, and push with an explicit `--force-with-lease` expectation.
Never publish backups or private audit reports. Check the remote branch after pushing.
