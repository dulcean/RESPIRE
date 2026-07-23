#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-SongGeneration}
CKPT=${CKPT:-ckpt/songgeneration_base_full}
INPUT=${INPUT:-input.jsonl}
OUT=${OUT:-output}

if [ ! -d "$REPO_DIR" ]; then
  git clone "${SONGGEN_REPO:-https://huggingface.co/tencent/SongGeneration}" "$REPO_DIR"
fi
cd "$REPO_DIR"
mkdir -p ckpt
if [ ! -d "$CKPT" ]; then
  huggingface-cli download tencent/SongGeneration --local-dir ckpt --include "songgeneration_base_full/*"
fi

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt || pip install -r requirements_nodeps.txt

sh generate.sh "$CKPT" "../$INPUT" "../$OUT" --low_mem

echo "Done. Audio in ../$OUT/audio/*.flac"
echo "Next (back on the metric box):"
echo "  uv run python generate/build_manifest.py --dir $OUT/audio \\"
echo "      --source generated:levo2 --out pilot/data/manifest_gen.csv"
