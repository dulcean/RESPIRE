from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

VERSES = [
    ("Morning light across the room", "I trace the shape of what we knew"),
    ("City hums beneath the rain", "Every window holds a name"),
    ("Paper boats along the stream", "Carry pieces of a dream"),
    ("Empty roads and neon signs", "Counting all the borderlines"),
    ("Quiet hands and quiet eyes", "Learning how the silence lies"),
    ("Somewhere past the second star", "I remember who we are"),
    ("Winter settles on the pier", "Colder now but still I'm here"),
    ("Photographs go soft and gold", "Stories that we never told"),
    ("Static on a distant phone", "Finding out I'm not alone"),
    ("Footsteps fading down the hall", "Waiting for the last footfall"),
]

CHORUSES = [
    ("Hold the line, hold the line", "We are burning bright tonight"),
    ("Run to me, run to me", "Underneath the open sky"),
    ("Let it go, let it go", "Every river finds the sea"),
    ("Call my name, call my name", "I will always answer you"),
    ("Rise again, rise again", "Nothing here can hold us down"),
]

BRIDGES = [
    "And when the morning comes / I'll still be standing here",
    "Break it down to nothing / then we build it up again",
    "All the words I could not say / are echoing away",
]

GENDERS = ["female vocal", "male vocal"]
GENRES = ["pop", "indie rock", "synthpop", "acoustic folk", "r&b", "electronic pop"]
EMOTIONS = ["uplifting", "melancholic", "energetic", "dreamy", "nostalgic"]
TEMPI = ["mid tempo", "upbeat", "slow"]


def build_lyric(v_pair_1, v_pair_2, chorus, bridge) -> str:
    C = f"{chorus[0]}\n{chorus[1]}"
    return (
        "[intro-short]\n"
        f"[verse]\n{v_pair_1[0]}\n{v_pair_1[1]}\n"
        f"[chorus]\n{C}\n"
        f"[verse]\n{v_pair_2[0]}\n{v_pair_2[1]}\n"
        f"[chorus]\n{C}\n"
        f"[bridge]\n{bridge}\n"
        f"[chorus]\n{C}\n"
        "[outro-short]"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("generate/input.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    descs = list(itertools.product(GENDERS, GENRES, EMOTIONS, TEMPI))
    rng.shuffle(descs)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as f:
        for i in range(args.n):
            v1, v2 = rng.sample(VERSES, 2)
            chorus = rng.choice(CHORUSES)
            bridge = rng.choice(BRIDGES)
            g, genre, emo, tempo = descs[i % len(descs)]
            row = {
                "idx": f"gen_{i:04d}",
                "gt_lyric": build_lyric(v1, v2, chorus, bridge),
                "descriptions": f"{emo}, {genre}, {g}, {tempo}",
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.n} songs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
