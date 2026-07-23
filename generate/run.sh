#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-SongGeneration}
CKPT=${CKPT:-songgeneration_v2_large/model.pt}
INPUT=${INPUT:-input.jsonl}
OUT=${OUT:-output}

if [ ! -d "$REPO_DIR" ]; then
  uvx hf auth login
  #git clone "${SONGGEN_REPO:-https://huggingface.co/tencent/SongGeneration}" "$REPO_DIR"
  uvx hf download lglg666/SongGeneration-v2-large --local-dir ./songgeneration_v2_large
fi
cd "$REPO_DIR"
mkdir -p ckpt
if [ ! -d "$CKPT" ]; then
  uvx hf download tencent/SongGeneration --local-dir ckpt --include "songgeneration_base_full/*"
fi

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt || pip install -r requirements_nodeps.txt

sh generate.sh "$CKPT" "../$INPUT" "../$OUT" --low_mem

echo "Done. Audio in ../$OUT/audio/*.flac"
echo "Next (back on the metric box):"
echo "  uv run python generate/build_manifest.py --dir $OUT/audio \\"
echo "      --source generated:levo2 --out pilot/data/manifest_gen.csv"
