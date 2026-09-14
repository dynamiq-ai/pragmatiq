"""Realism reports: self-contained HTML plots plus machine-readable JSON checks.

Plots: events-per-user histogram, inter-event times,
amount-by-MCC, hour-of-day, merchant Zipf, label prevalences. Figures embed as
base64 PNGs so the HTML file has no external assets.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fig_to_b64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
    except ImportError as _e:
        from pragmatiq.core.errors import MissingExtraError
        raise MissingExtraError.for_extra("data", "matplotlib") from _e

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _check(name: str, value: float, threshold: float, op: str) -> dict[str, Any]:
    """One machine-readable realism check."""
    passed = value >= threshold if op == ">=" else value <= threshold
    return {"value": round(float(value), 6), "threshold": threshold, "op": op, "pass": bool(passed)}


def _calibration_residuals(manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Target-vs-realized residuals for calibrated moments present in the manifest."""
    cfg = manifest.get("config", {})
    prevalence = manifest.get("label_prevalence", {})
    pairs = {
        "default_rate": ("default_12m", "default_rate"),
        "fraud_user_rate": ("fraud_users", "fraud_rate"),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (actual_key, target_key) in pairs.items():
        if actual_key not in prevalence or target_key not in cfg:
            continue
        actual = float(prevalence[actual_key])
        target = float(cfg[target_key])
        out[name] = {
            "actual": round(actual, 6),
            "target": round(target, 6),
            "residual": round(actual - target, 6),
            "abs_residual": round(abs(actual - target), 6),
        }
    return out


def realism_metrics(agg: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable synthetic realism metrics collected during generation."""
    epu = np.asarray(agg.events_per_user, dtype=np.float64)
    p50 = float(np.percentile(epu, 50)) if epu.size else 0.0
    p95 = float(np.percentile(epu, 95)) if epu.size else 0.0
    day = float(np.asarray(agg.hour_hist)[[10, 12, 13, 17, 18]].mean())
    night = float(np.asarray(agg.hour_hist)[[1, 2, 3, 4]].mean())
    top = agg.merchant_counts.most_common(50)
    merchant_top1_rank50 = float(top[0][1] / max(top[-1][1], 1)) if len(top) >= 50 else 0.0
    mcc_medians = [
        float(np.median(v)) for v in agg.amounts_by_mcc.values() if len(v) >= 5
    ]
    amount_median_range = (
        float(max(mcc_medians) / max(min(mcc_medians), 1e-6)) if len(mcc_medians) >= 2 else 0.0
    )
    calibration = _calibration_residuals(manifest)
    default_resid = calibration.get("default_rate", {}).get("abs_residual", 0.0)
    fraud_resid = calibration.get("fraud_user_rate", {}).get("abs_residual", 0.0)
    return {
        "n_users": int(manifest["n_users"]),
        "n_events": int(manifest["n_events"]),
        "n_transfers": int(manifest["n_transfers"]),
        "label_prevalence": manifest.get("label_prevalence", {}),
        "calibration_residuals": calibration,
        "events_per_user": {"p50": round(p50, 3), "p95": round(p95, 3), "mean": round(float(epu.mean()), 3)},
        "hour_hist": [int(x) for x in np.asarray(agg.hour_hist).tolist()],
        "source_counts": {str(k): int(v) for k, v in sorted(agg.source_counts.items())},
        "checks": {
            "events_per_user_long_tail": _check("events_per_user_long_tail", p95 / max(p50, 1.0), 1.5, ">="),
            "hour_day_night_structure": _check("hour_day_night_structure", day / max(night, 1.0), 2.0, ">="),
            "merchant_zipf_concentration": _check(
                "merchant_zipf_concentration", merchant_top1_rank50, 2.0, ">="
            ),
            "amounts_differ_by_mcc": _check("amounts_differ_by_mcc", amount_median_range, 2.0, ">="),
            "calibration_default_rate_residual": _check(
                "calibration_default_rate_residual", default_resid, 0.08, "<="
            ),
            "calibration_fraud_user_rate_residual": _check(
                "calibration_fraud_user_rate_residual", fraud_resid, 0.03, "<="
            ),
        },
    }


def write_realism_report(agg: Any, manifest: dict[str, Any], path: str | Path) -> None:
    """Render the aggregates collected during generation into an HTML report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as _e:
        from pragmatiq.core.errors import MissingExtraError
        raise MissingExtraError.for_extra("data", "matplotlib") from _e

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
· seed {manifest['config']['seed']}</p>
<p><em>pragmatiq is an independent implementation inspired by the PRAGMA paper
(arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.</em></p>
<h3>Label prevalence</h3><table><tr><th>task</th><th>positive rate</th></tr>{prev_rows}</table>
<h3>Events by source</h3><table><tr><th>source</th><th>events</th></tr>{src_rows}</table>
{sections}
</body></html>"""
    path = Path(path)
    path.write_text(html)
    path.with_suffix(".json").write_text(json.dumps(realism_metrics(agg, manifest), indent=2, sort_keys=True))
