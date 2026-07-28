#!/bin/bash
# OWL-ViT base-patch32 가중치 다운로드 — 불안정 네트워크 대응 (curl 이어받기, 단일 프로세스)
# 사용: bash scripts/download_owlvit.sh
set -u
DEST="models/owlvit-base-patch32"
BASE="https://huggingface.co/google/owlvit-base-patch32/resolve/main"
mkdir -p "$DEST"
FILES="config.json preprocessor_config.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json pytorch_model.bin"

for f in $FILES; do
  for attempt in 1 2 3 4 5 6 7 8; do
    echo "[$f] 시도 $attempt"
    curl -L -C - --retry 3 --retry-delay 5 --connect-timeout 30 -o "$DEST/$f" "$BASE/$f" && break
    sleep 10
  done
done

# 검증: pytorch_model.bin 크기 (~613MB = 6.1e8 바이트 이상)
size=$(stat -c %s "$DEST/pytorch_model.bin" 2>/dev/null || echo 0)
echo "pytorch_model.bin: $size bytes"
if [ "$size" -lt 600000000 ]; then
  echo "DOWNLOAD_INCOMPLETE"
  exit 1
fi
echo "DOWNLOAD_OK"
