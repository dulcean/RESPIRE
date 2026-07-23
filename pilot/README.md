# Stage 0 — Pilot

The analysis engine is built and validated (`src//`). This directory drives it.
Stage 0 is the mandatory gate before any architecture work (see `TASK.md`).

## What the pipeline does

For every song: MERT frame embeddings → self-similarity matrix → Foote-novelty
boundaries → agglomerative section labels → locate the **chorus group** (≥2 returns) →
score returns `chorus-1 \ chorus-2` and `chorus-1 \ chorus-3` on:

| facet | extractor | question |
|-------|-----------|----------|
| melody | MERT | did the tune come back? |
| vocal_timbre | WavLM-SV | is it the same voice? |
| arrangement | chroma | same harmonic skeleton? |
| audio | CLAP | overall audio match |

Report = distribution of recall for **generated vs real** + the **1-2 → 1-3 drift**.

## Populate data (target: ~200 generated + ~200 real, verse-chorus-verse-chorus)

Put files anywhere and list them in a manifest CSV (`path,source`):

```csv
path,source
/data/gen/levo2/0001.wav,generated:levo2
/data/real/mtg/0001.mp3,real:mtg-jamendo
```

`source` prefix must be `generated:<system>` or `real:<corpus>`.

**Generated songs** — run open backbones with inference only (no retraining):
- LeVo 2 / SongGeneration 2 (Tencent, open weights) — the target backbone.
- ACE-Step, YuE — for cross-system robustness.
Prompt them with lyrics that have a repeating chorus so a chorus group exists.

**Real songs** — open corpora with clear verse-chorus form:
- MTG-Jamendo, FMA (CC), or any licensed set. Full tracks, not clips.

## Run

```bash
uv run python pilot/run_pilot.py --manifest pilot/data/manifest.csv \
    --extractors mert clap muq speaker --out pilot --limit 5


uv run python pilot/run_pilot.py --manifest pilot/data/manifest.csv \
    --extractors mert clap muq speaker --out pilot
```

Outputs: `pilot/results.md`, `pilot/recall_distributions.png`, `pilot/recall_raw.csv`,
`pilot/summary.json`.
