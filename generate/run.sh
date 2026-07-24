

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=${ROOT:-"$HERE/SongGeneration"}
CFG=${CFG:-"$ROOT/conf/base_config.yaml"}
MODEL=${MODEL:-"$ROOT/ckpt/songgeneration_base/model.pt"}
INPUT=${INPUT:-"$HERE/input.jsonl"}
OUT=${OUT:-"$HERE/output"}
BUNDLE=${BUNDLE:-"$HERE/ckpt_bundle"}

if [ ! -d "$ROOT/ckpt/vae" ] && [ -d "$BUNDLE/ckpt/vae" ]; then
  echo "merging weights bundle into $ROOT ..."
  cp -rn "$BUNDLE/ckpt" "$ROOT/"
  mkdir -p "$ROOT/third_party"
  cp -rn "$BUNDLE"/third_party/* "$ROOT/third_party/"
fi


mkdir -p "$ROOT/custom_nodes/ComfyUI_SongGeneration"
ln -sfn "$ROOT" "$ROOT/custom_nodes/ComfyUI_SongGeneration/SongGeneration"

[ -f "$MODEL" ]           || { echo "no LM at $MODEL"; exit 1; }
[ -f "$CFG" ]             || { echo "no config at $CFG"; exit 1; }
[ -d "$ROOT/ckpt/vae" ]   || { echo "no decoder weights at $ROOT/ckpt/vae (download ModelScope bundle to $BUNDLE)"; exit 1; }
[ -f "$INPUT" ]           || { echo "missing $INPUT (run make_jsonl.py first)"; exit 1; }

uv pip install --quiet "transformers==4.40.2" "tokenizers>=0.19,<0.20"

uv pip install --quiet descript-audio-codec alias-free-torch vector-quantize-pytorch \
  x-transformers einops-exts local-attention k-diffusion ema-pytorch prefigure aeiou \
  openunmix lameenc julius


uv pip install --quiet "huggingface-hub>=0.23,<1.0" "setuptools<66"

uv run --no-sync python "$HERE/batch_generate.py" \
  --config "$CFG" --model "$MODEL" --weights "$ROOT" --version v1 \
  --input "$INPUT" --out "$OUT" --code "$ROOT"

echo "Done -> $OUT/audio/*.flac"
echo "Back on the metric box:"
echo "  uv run python generate/build_manifest.py --dir generate/output/audio \\"
echo "      --source generated:songgen-base --out pilot/data/manifest_gen.csv"
