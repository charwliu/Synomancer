#!/bin/zsh
# Prepare an offline HF cache export; weights remain symlinked to the original cache.
set -euo pipefail
if (( $# != 3 && $# != 4 )); then
  print -u2 'usage: prepare-qwen4-gguf.sh tokenizer-assets gguf-snapshot export-directory [complete-mtp-sidecar]'
  exit 2
fi
assets="${1:A}"
snapshot="${2:A}"
destination="${3:A}"
first='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
second='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf'
projector='mmproj-Qwen3.8-Flash-Next-BF16.gguf'
for name in config.json tokenizer.json tokenizer_config.json preprocessor_config.json; do
  [[ -s "$assets/$name" ]] || { print -u2 "Missing $assets/$name"; exit 1; }
done
for name in "IQ3_S/$first" "IQ3_S/$second" "$projector"; do
  [[ -s "$snapshot/$name" ]] || { print -u2 "Missing $snapshot/$name"; exit 1; }
done
if (( $# == 4 )); then
  [[ -s "$4" ]] || { print -u2 "Missing MTP sidecar: $4"; exit 1; }
fi
[[ ! -e "$destination" ]] || { print -u2 'Export directory already exists; refusing to overwrite it.'; exit 1; }
mkdir -p "$destination"
for name in config.json tokenizer.json tokenizer_config.json preprocessor_config.json; do
  cp "$assets/$name" "$destination/$name"
done
ln -s "$snapshot/IQ3_S" "$destination/IQ3_S"
ln -s "$snapshot/$projector" "$destination/$projector"
cat > "$destination/synomancer-gguf.json" <<'JSON'
{
  "firstShard": "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
  "projector": "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
}
JSON
if (( $# == 4 )); then
  cp "$4" "$destination/mtp-Qwen3.8-Flash-Next-Q8_0-complete.gguf"
  cat > "$destination/synomancer-gguf.json" <<'JSON'
{
  "firstShard": "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
  "projector": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "mtp": "mtp-Qwen3.8-Flash-Next-Q8_0-complete.gguf"
}
JSON
fi
print "Prepared native MLX model at $destination"
