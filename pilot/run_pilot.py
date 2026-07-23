from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from respire.audio import load_audio
from respire.embed import build_extractors
from respire.recall import analyze_song
from respire.report import write_report


def read_manifest(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--extractors", nargs="+", default=["mert", "clap", "muq", "speaker"]
    )
    ap.add_argument("--out", type=Path, default=Path("pilot"))
    ap.add_argument(
        "--limit", type=int, default=0, help="cap songs for a quick dry run"
    )
    args = ap.parse_args()

    entries = read_manifest(args.manifest)
    if args.limit:
        entries = entries[: args.limit]
    extractors = build_extractors(args.extractors)
    srs = sorted({e.sr for e in extractors.values()} | {22050})

    results = []
    for i, ent in enumerate(entries):
        p = Path(ent["path"])
        try:
            wav_by_sr = {sr: load_audio(p, sr) for sr in srs}
            r = analyze_song(
                p.stem, ent["source"], wav_by_sr, extractors, fps_for_seg=0.0
            )
            if r is None:
                print(
                    f"[{i+1}/{len(entries)}] {p.name}: no chorus group, skipped",
                    file=sys.stderr,
                )
                continue
            results.append(r)
            print(
                f"[{i+1}/{len(entries)}] {p.name}: {r.n_choruses} choruses ok",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[{i+1}/{len(entries)}] {p.name}: ERROR {e}", file=sys.stderr)

    out = write_report(results, args.out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
