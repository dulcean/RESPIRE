"""
  python generate/batch_generate.py --config generate/songgeneration_v2_large/config.yaml \
      --model generate/songgeneration_v2_large/model.pt --weights /path/to/ckpt_bundle \
      --input generate/input.jsonl --out generate/output --code generate/SongGeneration
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf


def register_resolvers(code_dir: Path) -> None:
    resolvers = {
        "eval": lambda x: eval(x),
        "concat": lambda *x: [z for y in x for z in y],
        "get_fname": lambda: "batch",
        "load_yaml": lambda x: list(OmegaConf.load(str(code_dir / x))),
    }
    for name, func in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, func)


def build_lm(cfg, model_path: str):
    from codeclm.models import builders

    audiolm = builders.get_lm_model(cfg, cfg.version, cfg.offload_audiolm)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = {
        k.replace("audiolm.", ""): v for k, v in ckpt.items() if k.startswith("audiolm")
    }
    del ckpt
    audiolm.load_state_dict(state, strict=False)
    del state
    gc.collect()
    return audiolm.eval().to(torch.float16)


def build_sep_tokenizer(cfg):
    from codeclm.models import builders

    return builders.get_audio_tokenizer_model_cpu(
        cfg.audio_tokenizer_checkpoint_sep, cfg
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="checkpoint config.yaml")
    ap.add_argument("--model", required=True, type=Path, help="audiolm model.pt")
    ap.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="aux bundle (vae/, model_septoken/, third_party/Qwen2-7B)",
    )
    ap.add_argument("--input", required=True, type=Path, help="input.jsonl")
    ap.add_argument("--out", type=Path, default=Path("generate/output"))
    ap.add_argument(
        "--code", required=True, type=Path, help="cloned SongGeneration code dir"
    )
    ap.add_argument("--max-duration", type=float, default=150.0)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--cfg-coef", type=float, default=1.5)
    args = ap.parse_args()

    sys.path.insert(0, str(args.code.resolve()))
    code_dir = args.code.resolve()
    register_resolvers(code_dir)

    from codeclm.models import CodecLM

    W = str(args.weights.resolve())
    cfg = OmegaConf.load(str(args.config))
    cfg.mode = "inference"
    cfg.version = getattr(cfg, "version", "v2")
    cfg.offload_audiolm = False
    # ModelScope/tencent bundle layout: vae + tokenizers under ckpt/, Qwen2-7B at root.
    cfg.vae_config = f"{W}/ckpt/vae/stable_audio_1920_vae.json"
    cfg.vae_model = f"{W}/ckpt/vae/autoencoder_music_1320k.ckpt"
    cfg.audio_tokenizer_checkpoint_sep = (
        f"Flow1dVAESeparate_{W}/ckpt/model_septoken/model_2.safetensors"
    )
    cfg.conditioners.type_info.QwTextTokenizer.token_path = f"{W}/third_party/Qwen2-7B"

    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    entries = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]

    audiolm = build_lm(cfg, str(args.model)).cuda()
    lm = CodecLM(
        name="tmp",
        lm=audiolm,
        audiotokenizer=None,
        max_duration=args.max_duration,
        seperate_tokenizer=None,
    )
    lm.set_generation_params(
        duration=args.max_duration,
        extend_stride=5,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=0.0,
        cfg_coef=args.cfg_coef,
        record_tokens=True,
        record_window=50,
    )

    for i, ent in enumerate(entries):
        idx = ent["idx"]
        out_path = args.out / "audio" / f"{idx}.flac"
        if out_path.exists():
            print(f"[{i+1}/{len(entries)}] {idx}: exists, skip", file=sys.stderr)
            continue
        lyric = ent["gt_lyric"].replace("  ", " ")
        desc = ent.get("descriptions", "")
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                tokens = lm.generate(
                    lyrics=[lyric],
                    descriptions=[desc],
                    melody_wavs=None,
                    vocal_wavs=None,
                    bgm_wavs=None,
                    melody_is_wav=True,
                    return_tokens=True,
                )

            sep = build_sep_tokenizer(cfg).eval().cuda()
            dec = CodecLM(
                name="tmp",
                lm=None,
                audiotokenizer=None,
                max_duration=args.max_duration,
                seperate_tokenizer=sep,
            )
            with torch.no_grad():
                wav = dec.generate_audio(tokens, chunked=True, gen_type="mixed")[0]
            sep = sep.cpu()
            del sep, dec
            gc.collect()
            torch.cuda.empty_cache()
            w = wav.detach().cpu().float()
            if w.ndim == 1:
                w = w.unsqueeze(0)
            torchaudio.save(str(out_path), w, cfg.sample_rate)
            print(f"[{i+1}/{len(entries)}] {idx}: ok", file=sys.stderr)
        except Exception as e:
            print(f"[{i+1}/{len(entries)}] {idx}: ERROR {e}", file=sys.stderr)

    print(f"Done -> {args.out}/audio/*.flac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
