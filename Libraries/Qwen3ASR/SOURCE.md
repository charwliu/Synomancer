# Source

Selected Qwen3-ASR, MLXCommon and AudioCommon sources from
https://github.com/soniqo/speech-swift at revision
`1f54e56cf137078ed681a03e0955e777f7314610` (Apache-2.0).

The package manifest links the workspace upstream MLX Swift 0.32.3 and
swift-transformers 1.3.4. Unused streaming/VAD, UI, server and other model
targets are omitted to avoid conflicting MLXLMCommon/MLXLLM products.
Model weights are downloaded at runtime and are not included in the repository.

Removed an unused mel-filter local variable to keep builds warning-free.
Use the current `quantizedMM` API instead of its deprecated alias.
Model loading does not change process-wide wired-memory or cache limits:
Synomancer also runs chat models on the same MLX runtime.

Synomancer additions: standard Hugging Face cache/snapshot resolution and
thread-safe PCM buffering/partial transcript state. BF16 decoder matrices
are quantized in memory to the decoder's 4bit format; source cache files are
never modified. Empty/mismatched weight namespaces fail with a loading error.

Qwen3TTS sources from the same pinned speech-swift revision are included for
local CustomVoice synthesis. Its cancellation-aware synthesis entry point
is public for the app's actor-owned generation task. Local loading uses
`wiredMemoryPolicy: .none` and does not download or modify cached weights.
HF cache validation follows snapshot symlinks to regular blob files.
Two unused local variables in upstream TTS sources were removed.

The app quantizes floating-point Talker and Code Predictor linear layers to
8bit in memory before warm-up. Embeddings, normalization and
speech codec retain their original precision. Cached weights are not rewritten.
Native Markdown answers are split into bounded text segments. The app consumes
`synthesizeStream` audio chunks into an AVAudioEngine PCM queue instead of WAV
files. A defaulted completion callback on the upstream streaming API lets the
actor wait for a cancelled GPU producer to exit before reusing its model.

Streaming codec continuity adaptation: decoder transformer attention K/V and
causal pre-convolution history now belong to each utterance. The waveform stack
retains at least 16 frames of true transformed latent history. Streaming and batch
chunked decode use the same sample clock, derived from actual waveform lengths;
startup is not padded with fabricated codec IDs. The convolution-only waveform
path remains compiled. These changes prevent sliding windows from restarting
attention every few words and avoid adding/dropping startup samples at seams.
The app uses 6/12-frame chunks, a 0.6-second playback prebuffer and approximately
four seconds of scheduled PCM plus one chunk. Playback metrics count buffer
underruns; tests use real local codec weights, synthetic codec parity, real
Chinese/English TTS-ASR and muted AVAudioEngine playback.
