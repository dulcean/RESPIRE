from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .recall import SongRecall

FACETS = ["melody", "vocal_timbre", "arrangement", "audio", "muq"]


def _group(source: str) -> str:
    return "real" if source.startswith("real") else "generated"


def to_records(results: list[SongRecall]) -> list[dict]:
    rows = []
    for r in results:
        for pair, facets in r.pairs.items():
            for facet, sim in facets.items():
                rows.append(
                    dict(
                        song=r.song_id,
                        group=_group(r.source),
                        source=r.source,
                        pair=pair,
                        facet=facet,
                        sim=sim,
                    )
                )
    return rows


def _stats(vals: list[float]) -> dict:
    a = np.array(vals, float)
    return dict(
        n=len(a), mean=float(a.mean()), std=float(a.std()), median=float(np.median(a))
    )


def write_report(results: list[SongRecall], out_dir: str | Path) -> Path:
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = to_records(results)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "recall_raw.csv", index=False)

    lines = ["# Stage 0 — Structural-Recall Pilot Results", ""]
    lines.append(
        f"Songs analysed: {len(results)} "
        f"(generated={sum(_group(r.source)=='generated' for r in results)}, "
        f"real={sum(_group(r.source)=='real' for r in results)})."
    )
    lines.append("")

    if df.empty:
        lines.append("**No chorus groups detected — populate data and rerun.**")
        (out_dir / "results.md").write_text("\n".join(lines))
        return out_dir / "results.md"

    lines.append("## Recall by group / facet / return-pair")
    lines.append("")
    lines.append("| facet | pair | generated (mean±std) | real (mean±std) | gap |")
    lines.append("|---|---|---|---|---|")
    drift = {}
    for facet in [f for f in FACETS if f in df.facet.unique()]:
        for pair in sorted(df.pair.unique()):
            sub = df[(df.facet == facet) & (df.pair == pair)]
            g = sub[sub.group == "generated"].sim
            r = sub[sub.group == "real"].sim
            if len(g) == 0 and len(r) == 0:
                continue
            gm = f"{g.mean():.3f}±{g.std():.3f}" if len(g) else "—"
            rm = f"{r.mean():.3f}±{r.std():.3f}" if len(r) else "—"
            gap = f"{(r.mean()-g.mean()):+.3f}" if len(g) and len(r) else "—"
            lines.append(f"| {facet} | {pair} | {gm} | {rm} | {gap} |")
            drift.setdefault(facet, {})[pair] = g.mean() if len(g) else np.nan
    lines.append("")

    lines.append("## Drift with length (generated songs: recall 1-2 vs 1-3)")
    lines.append("")
    verdict_drift = []
    for facet, pairs in drift.items():
        if (
            "1-2" in pairs
            and "1-3" in pairs
            and not np.isnan(pairs["1-2"])
            and not np.isnan(pairs["1-3"])
        ):
            d = pairs["1-2"] - pairs["1-3"]
            verdict_drift.append(d)
            lines.append(
                f"- **{facet}**: 1-2={pairs['1-2']:.3f}, 1-3={pairs['1-3']:.3f}, drop={d:+.3f}"
            )
    lines.append("")

    gen_lower = []
    for facet in FACETS:
        sub = df[df.facet == facet]
        g = sub[sub.group == "generated"].sim
        r = sub[sub.group == "real"].sim
        if len(g) and len(r):
            gen_lower.append(r.mean() - g.mean())
    lines.append("## Verdict")
    lines.append("")
    if gen_lower and np.mean(gen_lower) > 0.02:
        lines.append(
            f"- Generated recall is **lower** than real (mean gap {np.mean(gen_lower):+.3f}). "
            "Problem is present."
        )
    else:
        lines.append(
            "- No clear generated<real gap. **Hypothesis not supported — revisit.**"
        )
    if verdict_drift and np.mean(verdict_drift) > 0.01:
        lines.append(
            f"- Recall **drifts with length** (mean 1-2→1-3 drop {np.mean(verdict_drift):+.3f}). "
            "Supports the accumulation claim."
        )
    else:
        lines.append("- No consistent length drift detected.")
    lines.append("")
    lines.append(
        "**Decision:** CONTINUE if both a generated<real gap and a length drift hold; "
        "otherwise reconsider the problem statement."
    )

    (out_dir / "results.md").write_text("\n".join(lines))
    _plot(df, out_dir)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_songs": len(results),
                "gap_mean": float(np.mean(gen_lower)) if gen_lower else None,
                "drift_mean": float(np.mean(verdict_drift)) if verdict_drift else None,
            },
            indent=2,
        )
    )
    return out_dir / "results.md"


def _plot(df, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    facets = [f for f in FACETS if f in df.facet.unique()]
    fig, axes = plt.subplots(
        1, len(facets), figsize=(4 * len(facets), 4), squeeze=False
    )
    for ax, facet in zip(axes[0], facets):
        data, labels = [], []
        for grp in ["generated", "real"]:
            v = df[(df.facet == facet) & (df.group == grp)].sim.values
            if len(v):
                data.append(v)
                labels.append(grp)
        if data:
            ax.boxplot(data, tick_labels=labels)
        ax.set_title(facet)
        ax.set_ylabel("recall (cosine)")
    fig.suptitle("Stage 0: structural recall — generated vs real")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_distributions.png", dpi=120)
    plt.close(fig)
