#!/bin/bash
# CLIP ViT-B/32 (transformers, openai/clip-vit-base-patch32) 로컬 다운로드
# 팀원 scene_cuts와 동일 백본(ViT-B/32). 폐쇄망 추론 대비 로컬 로드용.
set -u
DEST="models/clip-vit-base-patch32"
BASE="https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"
mkdir -p "$DEST"
FILES="config.json preprocessor_config.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json pytorch_model.bin"
for f in $FILES; do
  for attempt in 1 2 3 4 5 6; do
    echo "[$f] 시도 $attempt"
    curl -L -C - --retry 3 --retry-delay 5 --connect-timeout 30 -o "$DEST/$f" "$BASE/$f" && break
    sleep 8
  done
done
size=$(stat -c %s "$DEST/pytorch_model.bin" 2>/dev/null || echo 0)
echo "pytorch_model.bin: $size bytes"
[ "$size" -gt 500000000 ] && echo "DOWNLOAD_OK" || { echo "DOWNLOAD_INCOMPLETE"; exit 1; }
