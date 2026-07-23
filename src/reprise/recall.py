from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embed import Extractor
from .segment import Section, chorus_group, segment_song


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def chroma_arrangement(wav: np.ndarray, sr: int) -> np.ndarray:

    import librosa

    chroma = librosa.feature.chroma_cqt(y=wav, sr=sr)
    return chroma.mean(axis=1)


@dataclass
class SongRecall:
    song_id: str
    source: str
    n_choruses: int
    pairs: dict[str, dict[str, float]] = field(default_factory=dict)


def _seg_wav(wav: np.ndarray, sr: int, sec: Section) -> np.ndarray:
    a = int(sec.start_s * sr)
    b = int(sec.end_s * sr)
    return wav[a:b]


def analyze_song(
    song_id: str,
    source: str,
    wav_by_sr: dict[int, np.ndarray],
    extractors: dict[str, Extractor],
    fps_for_seg: float,
) -> SongRecall | None:

    seg_ext = extractors.get("mert") or next(iter(extractors.values()))
    seg_wav = wav_by_sr[seg_ext.sr]
    frames = seg_ext.embed_frames(seg_wav, seg_ext.sr)
    fps = len(frames) / (len(seg_wav) / seg_ext.sr)
    sections = segment_song(frames, fps)
    chorus = chorus_group(sections)
    if not chorus or len(chorus) < 2:
        return None

    res = SongRecall(song_id=song_id, source=source, n_choruses=len(chorus))
    ref = chorus[0]
    for idx, other in enumerate(chorus[1:], start=2):
        key = f"1-{idx}"
        facets: dict[str, float] = {}

        for fname, ext in extractors.items():
            w = wav_by_sr[ext.sr]
            e_ref = ext.embed_segment(_seg_wav(w, ext.sr, ref), ext.sr)
            e_oth = ext.embed_segment(_seg_wav(w, ext.sr, other), ext.sr)
            facet = {
                "mert": "melody",
                "speaker": "vocal_timbre",
                "clap": "audio",
                "muq": "muq",
            }.get(fname, fname)
            facets[facet] = _cos(e_ref, e_oth)

        sr22 = 22050 if 22050 in wav_by_sr else seg_ext.sr
        w22 = wav_by_sr[sr22]
        facets["arrangement"] = _cos(
            chroma_arrangement(_seg_wav(w22, sr22, ref), sr22),
            chroma_arrangement(_seg_wav(w22, sr22, other), sr22),
        )
        res.pairs[key] = facets
    return res
