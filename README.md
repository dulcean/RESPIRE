# RESPIRE

Pointer-based structural recall for full-song generation. Choruses that return instead
of being re-sung — asymmetric section reuse over hierarchical audio tokens.

## Layout

```
src/reprise/     structural-recall metric engine
  audio.py       audio loading / device
  embed.py       MERT / CLAP / MuQ / speaker extractors
  segment.py     SSM + Foote-novelty segmentation, chorus-group detection
  recall.py      per-facet return-fidelity scoring
  report.py      aggregation, tables, drift check, plots
pilot/           Stage-0 runner + data instructions (pilot/README.md)
```

## Quick start

```bash
uv sync
uv run python pilot/run_pilot.py --manifest pilot/data/manifest.csv \
    --extractors mert clap muq speaker --out pilot
```

See `pilot/README.md` for data collection and the CONTINUE/revisit decision rule.
