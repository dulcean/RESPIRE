from __future__ import annotations

import argparse
import csv
from pathlib import Path

EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--source", required=True, help="e.g. generated:levo2 or real:mtg-jamendo")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--glob", default="*", help="filter, e.g. '*_[!v]*' to skip *_vocal")
    args = ap.parse_args()

    files = sorted(
        p for p in args.dir.rglob(args.glob)
        if p.suffix.lower() in EXTS and "_vocal" not in p.stem and "_bgm" not in p.stem
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, ["path", "source"])
        w.writeheader()
        for p in files:
            w.writerow({"path": str(p.resolve()), "source": args.source})
    print(f"{len(files)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
