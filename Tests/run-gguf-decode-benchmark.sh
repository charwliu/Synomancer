#!/bin/zsh
# Build first with Tests/run-gguf-tests.sh --kernels. No Python runtime.
set -euo pipefail
if (( $# != 4 )); then
  print -u2 'usage: run-gguf-decode-benchmark.sh first-shard.gguf mtp.gguf fixture.json resident|paged'
  exit 2
fi
root="${0:A:h:h}"
# Preserve HF snapshot symlink names: the loader derives shard 2 from shard 1.
shard="${1:a}"
export SYNOMANCER_GGUF_MTP="${2:a}"
fixture="${3:a}"
case "$4" in
  resident) export SYNOMANCER_GGUF_RESIDENT=1 ;;
  paged) export SYNOMANCER_GGUF_RESIDENT=0 SYNOMANCER_GGUF_EXPERT_GIB=16 ;;
  *) print -u2 'mode must be resident or paged'; exit 2 ;;
esac
# First four runs warm both paths and both generation modes. Then four samples
# per configuration, pairing baseline/optimized and reversing order on alternate rounds.
export SYNOMANCER_GGUF_PR9_ORDER=b0,o0,b1,o1,b0,o0,b1,o1,o0,b0,o1,b1,b0,o0,b1,o1,o0,b0,o1,b1
export SYNOMANCER_GGUF_PR9_PAIRED="${SYNOMANCER_GGUF_PR9_PAIRED:-1}"
cd "$root/.build/DerivedData/Build/Products/Release"
exec ./Qwen4GGUFSmoke "$shard" "$fixture"
