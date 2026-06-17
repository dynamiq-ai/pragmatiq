"""Phase 1b: calibrate generator priors to bank-shareable aggregates.

Banks never share raw data. ``calibrate_config`` takes a YAML of aggregate
statistics (see ``configs/data/aggregates.example.yaml``) and produces a
WorldConfig dict whose simulated book matches those aggregates via simple
moment matching:

- ``archetype_shares``  → archetype_mix (renormalized)
- ``fraud_base_rate``   → fraud_rate
- ``default_rate``      → default_rate
- ``mcc_mix``           → mcc_weights (merchant universe composition)
- ``mean_amount_by_mcc``→ mcc_amount_mean (lognormal mean matching)
- ``events_per_user_month`` → activity_scale, fitted with short pilot
  simulations (1–2 iterations of multiplicative correction)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import WorldConfig
from .simulator import UserSimulator
from .world import MCC_KEYS, World, user_rng

log = logging.getLogger(__name__)

_PILOT_USERS = 400


def _hist_mean(spec: Any) -> float:
    """Mean event rate from a scalar or an event-rate histogram.

    Accepts a scalar (used directly) or a ``{rate: count}`` / ``{rate: weight}``
    histogram, returning its weighted mean (first-moment match, per the spec's
    "simple moment matching").
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        rates = np.array([float(k) for k in spec], dtype=np.float64)
        weights = np.array([float(v) for v in spec.values()], dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError("events_per_user_month histogram has non-positive total weight")
        return float((rates * weights).sum() / weights.sum())
    raise ValueError(f"events_per_user_month must be scalar or histogram dict, got {type(spec)}")


def _pilot_events_per_user_month(cfg: WorldConfig) -> float:
    """Simulate a small pilot population and measure mean events/user/month."""
    pilot = WorldConfig.from_dict({**cfg.to_dict(), "n_users": _PILOT_USERS})
    world = World.build(pilot)
    sim = UserSimulator(world)
    total = 0
    for u in range(pilot.n_users):
        rng = user_rng(pilot.seed, u)
        total += len(sim.run(u, rng).ts)
    return total / pilot.n_users / pilot.months


def calibrate_config(stats: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Moment-match a WorldConfig to aggregate statistics.

    Args:
        stats: parsed aggregates YAML (unknown keys are reported and ignored).
        base: starting WorldConfig fields (defaults if ``None``).

    Returns:
        A WorldConfig-compatible dict (also validated by construction).
    """
    out: dict[str, Any] = dict(base or {})
    known = {
        "archetype_shares", "fraud_base_rate", "default_rate", "mcc_mix",
        "mean_amount_by_mcc", "events_per_user_month", "n_users", "months", "country_mix",
    }
    ignored = set(stats) - known
    if ignored:
        log.warning("calibrate: ignoring unknown aggregate keys: %s", sorted(ignored))

    if "archetype_shares" in stats:
        shares = {str(k): float(v) for k, v in stats["archetype_shares"].items()}
        total = sum(shares.values())
        out["archetype_mix"] = {k: v / total for k, v in shares.items()}
    if "fraud_base_rate" in stats:
        out["fraud_rate"] = float(stats["fraud_base_rate"])
    if "default_rate" in stats:
        out["default_rate"] = float(stats["default_rate"])
    if "country_mix" in stats:
        cm = {str(k): float(v) for k, v in stats["country_mix"].items()}
        total = sum(cm.values())
        out["country_mix"] = {k: v / total for k, v in cm.items()}
    if "mcc_mix" in stats:
        mix = {str(k): float(v) for k, v in stats["mcc_mix"].items()}
        unknown = set(mix) - set(MCC_KEYS)
        if unknown:
            raise ValueError(f"mcc_mix has unknown MCC keys: {sorted(unknown)}; known: {list(MCC_KEYS)}")
        out["mcc_weights"] = mix
    if "mean_amount_by_mcc" in stats:
        means = {str(k): float(v) for k, v in stats["mean_amount_by_mcc"].items()}
        unknown = set(means) - set(MCC_KEYS)
        if unknown:
            raise ValueError(f"mean_amount_by_mcc has unknown MCC keys: {sorted(unknown)}")
        out["mcc_amount_mean"] = means
    for k in ("n_users", "months"):
        if k in stats:
            out[k] = int(stats[k])

    # Validate everything set so far.
    cfg = WorldConfig.from_dict(out)

    if "events_per_user_month" in stats:
        target = _hist_mean(stats["events_per_user_month"])
        for _ in range(2):  # multiplicative correction, 2 passes converge
            measured = _pilot_events_per_user_month(cfg)
            if measured <= 0:
                break
            scale = float(np.clip(cfg.activity_scale * target / measured, 0.05, 20.0))
            out["activity_scale"] = round(scale, 4)
            cfg = WorldConfig.from_dict(out)
            if abs(measured - target) / target < 0.07:
                break
        log.info("calibrate: activity_scale=%.3f (target %.1f events/user/month)",
                 cfg.activity_scale, target)

    return cfg.to_dict()
