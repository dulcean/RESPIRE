# Data generation

Two corpora feed the pilot metric: **generated** (LeVo 2) and **real** (MTG-Jamendo).

## 1. Generated songs (LeVo 2 / SongGeneration 2)

The T4 has 16 GB → use `songgeneration_base_full` + `--low_mem` (the *large* models
need 22–28 GB and won't fit). ~200 songs is an overnight batch.

```bash
uv run python generate/make_jsonl.py --n 200 --out generate/input.jsonl

huggingface-cli login
bash run.sh

uv run python generate/build_manifest.py --dir /path/to/output/audio \
    --source generated:levo2 --out pilot/data/manifest_gen.csv
```

*Notes*:
- For Stage 2 (track-asymmetric analysis) rerun with `--separate` to also get
  `{idx}_vocal.flac` / `{idx}_bgm.flac`. Stage 0 only needs the mixed track.

## 2. Real songs (MTG-Jamendo)

Full audio is ~500 GB (100 tar shards); we only need ~200 vocal/pop tracks.

```bash
git clone https://github.com/MTG/mtg-jamendo-dataset
uv run python data/mtg_sample.py --meta mtg-jamendo-dataset/data/raw_30s.tsv --n 200

python mtg-jamendo-dataset/scripts/download/download.py \
    --dataset raw_30s --type audio --output /data/mtg/audio

uv run python data/mtg_sample.py --meta mtg-jamendo-dataset/data/raw_30s.tsv --n 200 \
    --audio-root /data/mtg/audio --out pilot/data/manifest_real.csv
```

## 3. Run the metric on both

```bash
cat pilot/data/manifest_gen.csv > pilot/data/manifest.csv
tail -n +2 pilot/data/manifest_real.csv >> pilot/data/manifest.csv
uv run python pilot/run_pilot.py --manifest pilot/data/manifest.csv \
    --extractors mert clap muq speaker --out pilot
```