from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

WANT_TAGS = {"vocal", "pop", "rock", "singer", "female", "male", "melodic"}


def read_meta(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for r in reader:
            if len(r) < 5:
                continue
            rows.append(
                {
                    "track_id": r[0],
                    "path": r[3],
                    "duration": float(r[4]) if r[4].replace(".", "").isdigit() else 0.0,
                    "tags": {t.split("---")[-1].lower() for t in r[5:]},
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, type=Path, help="MTG metadata TSV")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="if set, only keep tracks whose audio file exists, and write manifest",
    )
    ap.add_argument("--out", type=Path, default=Path("pilot/data/manifest_real.csv"))
    ap.add_argument("--min-dur", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_meta(args.meta)
    cand = [
        r for r in rows if r["duration"] >= args.min_dur and (r["tags"] & WANT_TAGS)
    ]
    random.Random(args.seed).shuffle(cand)

    shards_needed = set()
    picked = []
    for r in cand:
        rel = r["path"]
        shard = rel.split("/")[0]
        if args.audio_root:
            ap_file = args.audio_root / rel
            if not ap_file.exists():
                continue
            r["_abs"] = ap_file
        picked.append(r)
        shards_needed.add(shard)
        if len(picked) >= args.n:
            break

    print(f"candidates={len(cand)} picked={len(picked)}")
    print("shards needed:", ",".join(sorted(shards_needed)) or "(none matched on disk)")

    if args.audio_root:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, ["path", "source"])
            w.writeheader()
            for r in picked:
                w.writerow(
                    {"path": str(r["_abs"].resolve()), "source": "real:mtg-jamendo"}
                )
        print(f"wrote manifest -> {args.out}")
    else:
        print(
            "Re-run with --audio-root once the listed shards are downloaded to build the manifest."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
