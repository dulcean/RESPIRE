from __future__ import annotations

from pathlib import Path

import numpy as np


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_audio(path: str | Path, target_sr: int, mono: bool = True) -> np.ndarray:

    import librosa

    y, _ = librosa.load(str(path), sr=target_sr, mono=mono)
    return y.astype(np.float32)
