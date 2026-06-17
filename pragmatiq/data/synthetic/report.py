"""Realism report: one self-contained HTML with the realism diagnostic plots.

Plots: events-per-user histogram, inter-event times,
amount-by-MCC, hour-of-day, merchant Zipf, label prevalences. Figures embed as
base64 PNGs so the file has no external assets.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np


def _fig_to_b64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_realism_report(agg: Any, manifest: dict[str, Any], path: str | Path) -> None:
    """Render the aggregates collected during generation into an HTML report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs: list[tuple[str, str]] = []

    epu = np.asarray(agg.events_per_user)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(epu, bins=60, color="#3b6ea5")
    ax.set_yscale("log")
    ax.set_title(f"Events per user (mean {epu.mean():.0f}, p99 {np.percentile(epu, 99):.0f})")
    ax.set_xlabel("events")
    imgs.append(("Events per user (long tail expected)", _fig_to_b64(fig)))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(np.arange(24), agg.hour_hist, color="#3b6ea5")
    ax.set_title("Hour-of-day histogram (day/night structure expected)")
    ax.set_xlabel("hour")
    imgs.append(("Hour of day", _fig_to_b64(fig)))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    centers = (np.arange(45) / 5.0) - 1
    ax.bar(centers, agg.delta_log10s_hist, width=0.18, color="#3b6ea5")
    ax.set_title("Inter-event time, log10(seconds)")
    ax.set_xlabel("log10 Δt (s)")
    imgs.append(("Inter-event times", _fig_to_b64(fig)))

    top = agg.merchant_counts.most_common(2000)
    if top:
        counts = np.array([c for _, c in top], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.loglog(np.arange(1, len(counts) + 1), counts, ".", ms=3, color="#3b6ea5")
        ax.set_title("Merchant frequency vs rank (≈ Zipf expected)")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
        imgs.append(("Merchant Zipf", _fig_to_b64(fig)))

    if agg.amounts_by_mcc:
        mccs = sorted(agg.amounts_by_mcc, key=lambda m: -len(agg.amounts_by_mcc[m]))[:12]
        data = [np.log10(np.clip(np.asarray(agg.amounts_by_mcc[m]), 0.01, None)) for m in mccs]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.boxplot(data, tick_labels=mccs, showfliers=False)
        ax.set_title("log10 amount by MCC (distributions must differ)")
        ax.tick_params(axis="x", rotation=45)
        imgs.append(("Amounts by MCC", _fig_to_b64(fig)))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(np.arange(7), agg.dow_hist, color="#3b6ea5")
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title("Day-of-week histogram")
    imgs.append(("Day of week", _fig_to_b64(fig)))

    prev_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in sorted(manifest.get("label_prevalence", {}).items())
    )
    src_rows = "".join(
        f"<tr><td>{k}</td><td>{v:,}</td></tr>" for k, v in sorted(agg.source_counts.items())
    )
    sections = "".join(
        f"<h3>{title}</h3><img src='data:image/png;base64,{b64}'/>" for title, b64 in imgs
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>pragmatiq synthetic data — realism report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:880px;margin:2em auto;color:#222}}
table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:4px 10px}}</style></head><body>
<h1>Realism report</h1>
<p>{manifest['n_users']:,} users · {manifest['n_events']:,} events · {manifest['n_transfers']:,} transfers
· seed {manifest['config']['seed']} · generated in {manifest['elapsed_sec']}s
({manifest['users_per_sec']} users/s)</p>
<p><em>pragmatiq is an independent implementation inspired by the PRAGMA paper
(arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.</em></p>
<h3>Label prevalence</h3><table><tr><th>task</th><th>positive rate</th></tr>{prev_rows}</table>
<h3>Events by source</h3><table><tr><th>source</th><th>events</th></tr>{src_rows}</table>
{sections}
</body></html>"""
    Path(path).write_text(html)
