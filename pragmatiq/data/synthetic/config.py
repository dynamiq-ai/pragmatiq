"""Configuration for the synthetic world generator.

Every knob that shapes the simulated book lives here so that calibration
(phase 1b) and tests can override behavior without touching simulator code.
All randomness downstream derives from ``seed`` (global rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_COUNTRY_MIX: dict[str, float] = {
    "GB": 0.55,
    "IE": 0.10,
    "FR": 0.10,
    "DE": 0.10,
    "ES": 0.05,
    "PL": 0.05,
    "LT": 0.05,
}

DEFAULT_ARCHETYPE_MIX: dict[str, float] = {
    "student": 0.10,
    "salaried": 0.38,
    "freelancer": 0.10,
    "family": 0.16,
    "pensioner": 0.07,
    "high_net_worth": 0.03,
    "trader": 0.05,
    "dormant": 0.06,
    "mule": 0.01,
    "fraud_victim": 0.04,
}


@dataclass
class WorldConfig:
    """Parameters of the simulated banking world.

    Attributes mirror Phase 1. ``trait_noise`` injects
    trait/behavior overlap so downstream tasks are realistically hard
    (gate 1 requires credit GBDT AUC in ~[0.75, 0.85], not >0.95).
    """

    n_users: int = 10_000
    start_date: str = "2023-01-01"
    months: int = 25
    country_mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COUNTRY_MIX))
    n_merchants: int = 50_000
    fraud_rate: float = 0.004  # share of users hit by a FraudEpisode
    default_rate: float = 0.03  # target default_12m prevalence among scored users
    mule_ring_count: int = 8
    seed: int = 0

    archetype_mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ARCHETYPE_MIX))

    # Realism / difficulty knobs (tuned at gate 1; see tests/baselines).
    trait_noise: float = 0.55  # 0 = labels fully determined by traits; 1 = mostly noise
    label_noise: float = 0.005  # random flip prob on binary user-level labels
    activity_scale: float = 1.0  # global multiplier on event volume
    # Fraction of event field entries randomly omitted from the events map, to
    # mirror the missing/sparse fields of real bank feeds. 0.0 (default) emits
    # complete fields and is byte-identical to a run without the knob. The
    # omission is a deterministic function of (seed, user, event ts, field), so it
    # stays reproducible and worker-count invariant, and exercises the absent-field
    # / [UNK] handling a bring-your-own-data pipeline relies on.
    missing_field_rate: float = 0.0

    # AML signal placement. Heavy-tailed organic P2P degree (sociability-driven
    # configuration model) keeps ring fan-in from being a structural oracle;
    # mule_behavior_strength scales the behavioral fingerprint (cash-out
    # bursts, dormancy arc) that embeddings — not graph degree — can detect.
    organic_degree_dispersion: float = 10.0  # mean extra stubs at sociability=1
    # 0.0 = relational regime: mules have NO distinctive individual cash-out
    # fingerprint, so AML signal lives purely in the transfer-graph ring (the
    # phase-6 relational-recovery target). >0 adds an embedding-visible ATM/crypto
    # cash-out burst scaled by the value (a behaviorally-mixed regime).
    mule_behavior_strength: float = 0.0  # 0 = graph-only AML signal

    # Label eval points, in months from start (lookahead must fit the horizon).
    eval_month_credit: int = 13  # default_12m: needs 12 months of lookahead
    eval_month_short: int = 19  # churn_6m / ltv_positive: 6 months of lookahead

    # Comms campaigns per month (each stores both potential outcomes for uplift).
    campaigns_per_month: float = 2.0

    # Calibration overrides (phase 1b): replace catalog MCC mix / amount means.
    # Keys are MCC catalog keys (e.g. "grocery"); validated at world build.
    mcc_weights: dict[str, float] | None = None
    mcc_amount_mean: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.months < 14:
            raise ValueError("months must be >= 14 (12m credit lookahead + history)")
        if not 0 <= self.trait_noise <= 1:
            raise ValueError("trait_noise must be in [0, 1]")
        if not 0 <= self.missing_field_rate < 1:
            raise ValueError("missing_field_rate must be in [0, 1)")
        if self.eval_month_credit + 12 > self.months:
            raise ValueError(
                f"eval_month_credit={self.eval_month_credit} needs 12 months of lookahead "
                f"but horizon is {self.months} months"
            )
        if self.eval_month_short + 6 > self.months:
            raise ValueError(
                f"eval_month_short={self.eval_month_short} needs 6 months of lookahead "
                f"but horizon is {self.months} months"
            )
        for name, mix in (("country_mix", self.country_mix), ("archetype_mix", self.archetype_mix)):
            total = sum(mix.values())
            if abs(total - 1.0) > 1e-6:
                # Normalize mixes within 5% of 1.0; reject wildly wrong ones.
                if abs(total - 1.0) > 0.05:
                    raise ValueError(f"{name} sums to {total:.3f}, expected 1.0")
                for k in mix:
                    mix[k] = mix[k] / total

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldConfig:
        """Build a config from a plain dict (e.g. an OmegaConf-resolved YAML)."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown WorldConfig keys: {sorted(unknown)}; known: {sorted(known)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (for run metadata and YAML round-trips)."""
        from dataclasses import asdict

        return asdict(self)
