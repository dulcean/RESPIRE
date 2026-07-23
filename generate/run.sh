#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT=${CKPT:-"$HERE/songgeneration_v2_large"}
INPUT=${INPUT:-"$HERE/input.jsonl"}
OUT=${OUT:-"$HERE/output"}
CODE_DIR=${CODE_DIR:-"$HERE/SongGeneration"}
WEIGHTS=${WEIGHTS:-"$HERE/ckpt_bundle"}

[ -f "$CKPT/model.pt" ]     || { echo "no checkpoint at $CKPT/model.pt"; exit 1; }
[ -f "$CKPT/config.yaml" ]  || { echo "no config at $CKPT/config.yaml"; exit 1; }
[ -f "$INPUT" ]             || { echo "missing $INPUT (run make_jsonl.py first)"; exit 1; }

if [ ! -d "$CODE_DIR" ]; then
  git clone --depth 1 "${SONGGEN_REPO:-https://github.com/smthemex/ComfyUI_SongGeneration}" "$HERE/_mirror"
  mv "$HERE/_mirror/SongGeneration" "$CODE_DIR"
  rm -rf "$HERE/_mirror"
fi
#
#
#
if [ ! -d "$WEIGHTS/ckpt/vae" ]; then
  echo "Aux decoder bundle not found at $WEIGHTS (need ckpt/vae, ckpt/model_septoken, third_party/Qwen2-7B)."
  echo "tencent/SongGeneration is gated/removed on HF; use the open ModelScope mirror:"
  echo "  uvx --from modelscope modelscope download AI-ModelScope/SongGeneration --local_dir $WEIGHTS"
  echo "  # (the LM you already have; you can skip ckpt/songgeneration_base/model.pt, ~11GB)"
  exit 1
fi

uv sync --quiet

uv run python "$HERE/batch_generate.py" \
  --config "$CKPT/config.yaml" --model "$CKPT/model.pt" --weights "$WEIGHTS" \
  --input "$INPUT" --out "$OUT" --code "$CODE_DIR"

echo "Done -> $OUT/audio/*.flac"
echo "Back on the metric box:"
echo "  uv run python generate/build_manifest.py --dir generate/output/audio \\"
echo "      --source generated:levo2 --out pilot/data/manifest_gen.csv"
