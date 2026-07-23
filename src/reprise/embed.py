from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import pick_device


def _l2(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


@dataclass
class Extractor:
    name: str
    sr: int
    _model: object = None
    _proc: object = None

    def _load(self):
        raise NotImplementedError

    def embed_frames(self, wav: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError

    def embed_segment(self, wav: np.ndarray, sr: int) -> np.ndarray:
        frames = self.embed_frames(wav, sr)
        if frames.ndim == 1:
            return _l2(frames)
        return _l2(frames.mean(axis=0))


class MERTExtractor(Extractor):
    HF_ID = "m-a-p/MERT-v1-95M"

    def __init__(self):
        super().__init__(name="mert", sr=24000)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self._proc = Wav2Vec2FeatureExtractor.from_pretrained(
            self.HF_ID, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(self.HF_ID, trust_remote_code=True)
        self._model.eval().to(pick_device())
        self._torch = torch

    def embed_frames(self, wav: np.ndarray, sr: int) -> np.ndarray:
        self._load()
        torch = self._torch
        dev = pick_device()
        inputs = self._proc(wav, sampling_rate=self.sr, return_tensors="pt")
        with torch.no_grad():
            out = self._model(inputs["input_values"].to(dev), output_hidden_states=True)

        hs = torch.stack(out.hidden_states, dim=0).mean(0).squeeze(0)
        return _l2(hs.float().cpu().numpy())


class CLAPExtractor(Extractor):
    HF_ID = "laion/clap-htsat-unfused"

    def __init__(self):
        super().__init__(name="clap", sr=48000)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        self._proc = ClapProcessor.from_pretrained(self.HF_ID)
        self._model = ClapModel.from_pretrained(self.HF_ID).eval().to(pick_device())
        self._torch = torch

    def embed_frames(self, wav: np.ndarray, sr: int) -> np.ndarray:

        self._load()
        torch = self._torch
        dev = pick_device()
        win = self.sr * 4
        hop = self.sr * 2
        if len(wav) < win:
            wav = np.pad(wav, (0, win - len(wav)))
        embs = []
        for start in range(0, max(1, len(wav) - win + 1), hop):
            chunk = wav[start : start + win]
            inp = self._proc(audios=chunk, sampling_rate=self.sr, return_tensors="pt")
            with torch.no_grad():
                e = self._model.get_audio_features(
                    **{k: v.to(dev) for k, v in inp.items()}
                )
            embs.append(e.squeeze(0).float().cpu().numpy())
        return _l2(np.stack(embs))


class SpeakerExtractor(Extractor):

    HF_ID = "microsoft/wavlm-base-plus-sv"

    def __init__(self):
        super().__init__(name="speaker", sr=16000)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._proc = AutoFeatureExtractor.from_pretrained(self.HF_ID)
        self._model = (
            WavLMForXVector.from_pretrained(self.HF_ID).eval().to(pick_device())
        )
        self._torch = torch

    def embed_frames(self, wav: np.ndarray, sr: int) -> np.ndarray:
        self._load()
        torch = self._torch
        dev = pick_device()
        win = self.sr * 3
        hop = self.sr * 3
        if len(wav) < win:
            wav = np.pad(wav, (0, win - len(wav)))
        embs = []
        for start in range(0, max(1, len(wav) - win + 1), hop):
            chunk = wav[start : start + win]
            inp = self._proc(
                chunk, sampling_rate=self.sr, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                e = self._model(**{k: v.to(dev) for k, v in inp.items()}).embeddings
            embs.append(e.squeeze(0).float().cpu().numpy())
        return _l2(np.stack(embs))


class MuQExtractor(MERTExtractor):

    HF_ID = "OpenMuQ/MuQ-large-msd-iter"

    def __init__(self):
        Extractor.__init__(self, name="muq", sr=24000)

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from muq import MuQ

            self._model = MuQ.from_pretrained(self.HF_ID).eval().to(pick_device())
            self._torch = torch
            self._native = True
        except Exception:

            self._native = False
            super()._load()

    def embed_frames(self, wav: np.ndarray, sr: int) -> np.ndarray:
        self._load()
        if not getattr(self, "_native", False):
            return super().embed_frames(wav, sr)
        torch = self._torch
        dev = pick_device()
        x = torch.from_numpy(wav).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            out = self._model(x, output_hidden_states=True)
        hs = out.last_hidden_state.squeeze(0)
        return _l2(hs.float().cpu().numpy())


EXTRACTORS = {
    "mert": MERTExtractor,
    "clap": CLAPExtractor,
    "muq": MuQExtractor,
    "speaker": SpeakerExtractor,
}


def build_extractors(names: list[str]) -> dict[str, Extractor]:
    return {n: EXTRACTORS[n]() for n in names}
