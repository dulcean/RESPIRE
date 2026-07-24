#!/usr/bin/env bash
# Generate ~200 chorus-repeating songs with SongGeneration (base v1) on the T4.
# Everything lives under generate/SongGeneration/: codeclm code + conf/ + the ckpt/ and
# third_party/ weights (merged from the ModelScope bundle). Base v1 fits the T4 (~10GB).
# Run from the repo root:  bash generate/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=${ROOT:-"$HERE/SongGeneration"}              # code + weights, one tree
CFG=${CFG:-"$ROOT/conf/base_config.yaml"}
MODEL=${MODEL:-"$ROOT/ckpt/songgeneration_base/model.pt"}
INPUT=${INPUT:-"$HERE/input.jsonl"}
OUT=${OUT:-"$HERE/output"}
BUNDLE=${BUNDLE:-"$HERE/ckpt_bundle"}             # ModelScope download, merged in on first run

# One-time: fold the downloaded weights into the code tree so paths line up.
if [ ! -d "$ROOT/ckpt/vae" ] && [ -d "$BUNDLE/ckpt/vae" ]; then
  echo "merging weights bundle into $ROOT ..."
  cp -rn "$BUNDLE/ckpt" "$ROOT/"
  mkdir -p "$ROOT/third_party"
  cp -rn "$BUNDLE"/third_party/* "$ROOT/third_party/"
fi

[ -f "$MODEL" ]           || { echo "no LM at $MODEL"; exit 1; }
[ -f "$CFG" ]             || { echo "no config at $CFG"; exit 1; }
[ -d "$ROOT/ckpt/vae" ]   || { echo "no decoder weights at $ROOT/ckpt/vae (download ModelScope bundle to $BUNDLE)"; exit 1; }
[ -f "$INPUT" ]           || { echo "missing $INPUT (run make_jsonl.py first)"; exit 1; }

uv sync --quiet

uv run python "$HERE/batch_generate.py" \
  --config "$CFG" --model "$MODEL" --weights "$ROOT" --version v1 \
  --input "$INPUT" --out "$OUT" --code "$ROOT"

echo "Done -> $OUT/audio/*.flac"
echo "Back on the metric box:"
echo "  uv run python generate/build_manifest.py --dir generate/output/audio \\"
echo "      --source generated:songgen-base --out pilot/data/manifest_gen.csv"
