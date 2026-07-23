from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Section:
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    label: int


def self_similarity(frames: np.ndarray) -> np.ndarray:

    return frames @ frames.T


def _gaussian_checkerboard(size: int) -> np.ndarray:
    r = np.arange(-size, size + 1)
    x, y = np.meshgrid(r, r)
    g = np.exp(-0.02 * (x**2 + y**2))
    sign = np.sign(x) * np.sign(y)
    return g * sign


def foote_novelty(ssm: np.ndarray, kernel_size: int = 32) -> np.ndarray:
    k = _gaussian_checkerboard(kernel_size)
    n = ssm.shape[0]
    ks = 2 * kernel_size + 1
    pad = kernel_size
    padded = np.pad(ssm, pad, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        patch = padded[i : i + ks, i : i + ks]
        nov[i] = float((patch * k).sum())
    nov = np.maximum(nov, 0)
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def detect_boundaries(nov: np.ndarray, fps: float, min_section_s: float = 6.0) -> list[int]:
    smoothed = gaussian_filter1d(nov, sigma=2)
    min_dist = int(min_section_s * fps)
    peaks, _ = find_peaks(smoothed, distance=max(1, min_dist), prominence=0.05)
    bounds = [0, *peaks.tolist(), len(nov)]
    return sorted(set(bounds))


def label_sections(frames: np.ndarray, bounds: list[int], n_clusters: int | None = None) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        segs.append(frames[a:b].mean(axis=0))
    segs = np.stack(segs)
    segs = segs / (np.linalg.norm(segs, axis=1, keepdims=True) + 1e-8)
    n = len(segs)
    if n == 1:
        return [0]
    k = n_clusters or max(2, min(n, round(n / 1.8)))
    k = min(k, n)
    clus = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return clus.fit_predict(segs).tolist()


def segment_song(frames: np.ndarray, fps: float, kernel_size: int = 32) -> list[Section]:
    ssm = self_similarity(frames)
    nov = foote_novelty(ssm, kernel_size=min(kernel_size, max(4, ssm.shape[0] // 4)))
    bounds = detect_boundaries(nov, fps)
    labels = label_sections(frames, bounds)
    out = []
    for (a, b), lab in zip(zip(bounds[:-1], bounds[1:]), labels):
        out.append(Section(a, b, a / fps, b / fps, lab))
    return out


def repeated_groups(sections: list[Section]) -> dict[int, list[Section]]:
    groups: dict[int, list[Section]] = {}
    for s in sections:
        groups.setdefault(s.label, []).append(s)
    return {lab: segs for lab, segs in groups.items() if len(segs) >= 2}


def chorus_group(sections: list[Section]) -> list[Section] | None:

    groups = repeated_groups(sections)
    if not groups:
        return None

    def score(segs: list[Section]) -> float:
        dur = np.mean([s.end_s - s.start_s for s in segs])
        span = max(s.start_s for s in segs) - min(s.start_s for s in segs)
        return dur * (1 + span)

    best = max(groups.values(), key=score)
    return sorted(best, key=lambda s: s.start_s)
