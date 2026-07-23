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
if [ ! -d "$WEIGHTS/vae" ]; then
  echo "Aux weights not found at $WEIGHTS."
  echo "The decoder bundle (vae/, model_septoken/, third_party/Qwen2-7B) is the GATED repo"
  echo "tencent/SongGeneration. Accept its license on HF, then:"
  echo "  uvx hf auth login"
  echo "  uvx hf download tencent/SongGeneration --local-dir $WEIGHTS"
  echo "  # China mirror (open): modelscope download AI-ModelScope/SongGeneration --local_dir $WEIGHTS"
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
