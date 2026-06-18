"""Personas: archetypes, latent traits, and vectorized population sampling.

Each user gets an archetype (config mixture) and eight latent traits drawn from
per-archetype Beta/LogNormal priors. Traits causally drive behavior in the
simulator and, through behavior, the labels — never the other way round.
``trait_noise`` blends each trait toward an archetype-independent draw so that
populations overlap and downstream tasks stay realistically hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

TRAITS: tuple[str, ...] = (
    "income_level",  # multiplier on archetype base income (LogNormal)
    "spend_propensity",
    "financial_stress",
    "tech_savviness",
    "risk_appetite",
    "sociability",
    "churn_hazard",
    "fraud_vulnerability",
)

# Per-archetype priors. Beta(a, b) for the bounded traits; income is
# LogNormal(ln(base), sigma) in GBP/month. base_txn_rate is card txns/month
# at full activity; session_rate is app sessions/month.
ARCHETYPES: dict[str, dict[str, Any]] = {
    "student": {
        "income_base": 750.0, "income_sigma": 0.35,
        "spend_propensity": (5, 2), "financial_stress": (3, 4), "tech_savviness": (6, 2),
        "risk_appetite": (3, 4), "sociability": (5, 2), "churn_hazard": (3, 5),
        "fraud_vulnerability": (3, 5),
        "base_txn_rate": 38.0, "session_rate": 14.0, "age": (18, 26),
        "salary_kind": "loan",  # termly student loan + parental top-ups
        "rent_prob": 0.75, "subs_lambda": 2.2,
        "mcc_tilt": {"fast_food": 2.2, "streaming": 1.8, "transport": 1.6, "gambling": 1.2, "grocery": 0.8},
    },
    "salaried": {
        "income_base": 2500.0, "income_sigma": 0.40,
        "spend_propensity": (4, 3), "financial_stress": (2.5, 5), "tech_savviness": (4, 3),
        "risk_appetite": (3, 5), "sociability": (4, 3), "churn_hazard": (2, 6),
        "fraud_vulnerability": (2, 6),
        "base_txn_rate": 48.0, "session_rate": 9.0, "age": (23, 60),
        "salary_kind": "monthly",
        "rent_prob": 0.55, "subs_lambda": 2.8,
        "mcc_tilt": {"restaurant": 1.3, "fuel": 1.3, "grocery": 1.2},
    },
    "freelancer": {
        "income_base": 2300.0, "income_sigma": 0.55,
        "spend_propensity": (4, 3), "financial_stress": (4, 4), "tech_savviness": (5, 2.5),
        "risk_appetite": (4, 4), "sociability": (4, 3), "churn_hazard": (3, 5),
        "fraud_vulnerability": (2.5, 5),
        "base_txn_rate": 44.0, "session_rate": 11.0, "age": (22, 55),
        "salary_kind": "invoices",  # 1–3 irregular credits/month
        "rent_prob": 0.60, "subs_lambda": 3.0,
        "mcc_tilt": {"electronics": 1.4, "restaurant": 1.2, "telecom": 1.3},
    },
    "family": {
        "income_base": 3100.0, "income_sigma": 0.40,
        "spend_propensity": (5, 2.5), "financial_stress": (3, 4), "tech_savviness": (3.5, 3.5),
        "risk_appetite": (2, 6), "sociability": (3.5, 3.5), "churn_hazard": (1.5, 7),
        "fraud_vulnerability": (2.5, 5),
        "base_txn_rate": 62.0, "session_rate": 7.0, "age": (28, 55),
        "salary_kind": "monthly",
        "rent_prob": 0.45, "subs_lambda": 3.4,
        "mcc_tilt": {"grocery": 1.8, "utilities": 1.5, "clothing": 1.3, "pharmacy": 1.3},
    },
    "pensioner": {
        "income_base": 1400.0, "income_sigma": 0.30,
        "spend_propensity": (3, 4), "financial_stress": (2.5, 5), "tech_savviness": (2, 6),
        "risk_appetite": (1.5, 7), "sociability": (2.5, 5), "churn_hazard": (1.5, 8),
        "fraud_vulnerability": (5, 3),  # phishing-prone
        "base_txn_rate": 22.0, "session_rate": 3.5, "age": (62, 88),
        "salary_kind": "pension",  # 1st of month
        "rent_prob": 0.25, "subs_lambda": 1.0,
        "mcc_tilt": {"pharmacy": 2.0, "grocery": 1.6, "fuel": 0.7, "fast_food": 0.5},
    },
    "high_net_worth": {
        "income_base": 11500.0, "income_sigma": 0.55,
        "spend_propensity": (5, 2), "financial_stress": (1.5, 8), "tech_savviness": (4, 3),
        "risk_appetite": (4.5, 3), "sociability": (4, 3), "churn_hazard": (2, 7),
        "fraud_vulnerability": (2, 6),
        "base_txn_rate": 75.0, "session_rate": 8.0, "age": (32, 68),
        "salary_kind": "monthly",
        "rent_prob": 0.15, "subs_lambda": 3.6,
        "mcc_tilt": {"restaurant": 1.7, "airline": 2.4, "hotel": 2.2, "jewelry": 2.0, "grocery": 0.8},
    },
    "trader": {
        "income_base": 3800.0, "income_sigma": 0.50,
        "spend_propensity": (4, 3), "financial_stress": (3, 4), "tech_savviness": (6.5, 1.5),
        "risk_appetite": (7, 1.5), "sociability": (3.5, 3.5), "churn_hazard": (3, 5),
        "fraud_vulnerability": (2, 6),
        "base_txn_rate": 45.0, "session_rate": 18.0, "age": (21, 50),
        "salary_kind": "monthly",
        "rent_prob": 0.55, "subs_lambda": 2.6,
        "mcc_tilt": {"electronics": 1.5, "fast_food": 1.3, "streaming": 1.3},
    },
    "dormant": {
        "income_base": 1500.0, "income_sigma": 0.45,
        "spend_propensity": (2, 6), "financial_stress": (3, 4), "tech_savviness": (3, 4),
        "risk_appetite": (2, 6), "sociability": (2, 6), "churn_hazard": (6, 2),
        "fraud_vulnerability": (3, 4),
        "base_txn_rate": 4.0, "session_rate": 1.2, "age": (20, 70),
        "salary_kind": "none",  # salary goes to their primary bank elsewhere
        "rent_prob": 0.05, "subs_lambda": 0.4,
        "mcc_tilt": {},
    },
    "mule": {
        # Behaviorally ordinary by design: real money mules are recruited normal
        # accounts. The mule signal lives in the transfer-graph ring, not in
        # per-user behaviour, so the priors match a generic salaried/freelancer
        # profile (slightly more sociable so ring connectivity is plausible) and
        # carry no individual fingerprint (low income, high stress, gambling/ATM
        # tilt) that an isolated embedding could read as a single-field AML tell.
        "income_base": 2100.0, "income_sigma": 0.45,
        "spend_propensity": (4, 3), "financial_stress": (3, 4), "tech_savviness": (4.5, 3),
        "risk_appetite": (3.5, 4), "sociability": (4.5, 3), "churn_hazard": (3, 5),
        "fraud_vulnerability": (3, 4),
        "base_txn_rate": 42.0, "session_rate": 9.0, "age": (19, 40),
        "salary_kind": "monthly",
        "rent_prob": 0.55, "subs_lambda": 2.6,
        "mcc_tilt": {},
    },
    "fraud_victim": {
        # Behaviorally ~salaried; what sets them apart is fraud_vulnerability.
        "income_base": 2350.0, "income_sigma": 0.40,
        "spend_propensity": (4, 3), "financial_stress": (3, 4.5), "tech_savviness": (2.5, 5),
        "risk_appetite": (3, 5), "sociability": (4, 3), "churn_hazard": (2.5, 5),
        "fraud_vulnerability": (7, 1.5),
        "base_txn_rate": 46.0, "session_rate": 7.0, "age": (30, 75),
        "salary_kind": "monthly",
        "rent_prob": 0.50, "subs_lambda": 2.4,
        "mcc_tilt": {"grocery": 1.2, "online_retail": 1.3},
    },
}

ARCHETYPE_NAMES: tuple[str, ...] = tuple(ARCHETYPES)


@dataclass
class PersonaTable:
    """Columnar persona/trait table for the whole population (phase A output).

    All arrays have length ``n_users``; ``traits`` maps trait name → float array.
    """

    archetype_idx: np.ndarray  # int16 index into ARCHETYPE_NAMES
    country: np.ndarray  # object (ISO-2 str)
    age: np.ndarray  # int16
    income_monthly: np.ndarray  # float64 GBP
    signup_day: np.ndarray  # int32 day offset from world start
    traits: dict[str, np.ndarray]

    def archetype_name(self, user_idx: int) -> str:
        """Archetype name for one user."""
        return ARCHETYPE_NAMES[int(self.archetype_idx[user_idx])]

    @property
    def n_users(self) -> int:
        return len(self.archetype_idx)


def sample_personas(cfg: Any, rng: np.random.Generator) -> PersonaTable:
    """Vectorized population draw (archetypes, traits, demographics, signup).

    ``trait_noise`` blends each Beta trait toward a population-level draw:
    ``t = (1-w)·t_archetype + w·t_anywhere`` which keeps marginals in [0, 1]
    while shrinking between-archetype separation.
    """
    n = cfg.n_users
    names = list(cfg.archetype_mix)
    for nm in names:
        if nm not in ARCHETYPES:
            raise ValueError(f"unknown archetype {nm!r} in archetype_mix; known: {ARCHETYPE_NAMES}")
    probs = np.array([cfg.archetype_mix[nm] for nm in names], dtype=np.float64)
    probs = probs / probs.sum()
    arch_in_mix = rng.choice(len(names), size=n, p=probs)
    # Map mixture order to canonical ARCHETYPE_NAMES order.
    canon = np.array([ARCHETYPE_NAMES.index(nm) for nm in names], dtype=np.int16)
    archetype_idx = canon[arch_in_mix]

    countries = list(cfg.country_mix)
    c_probs = np.array([cfg.country_mix[c] for c in countries], dtype=np.float64)
    c_probs = c_probs / c_probs.sum()
    country = np.array(countries, dtype=object)[rng.choice(len(countries), size=n, p=c_probs)]

    age = np.zeros(n, dtype=np.int16)
    income = np.zeros(n, dtype=np.float64)
    traits: dict[str, np.ndarray] = {t: np.zeros(n, dtype=np.float64) for t in TRAITS}
    w = float(cfg.trait_noise)

    for ai, nm in enumerate(ARCHETYPE_NAMES):
        mask = archetype_idx == ai
        m = int(mask.sum())
        if m == 0:
            continue
        spec = ARCHETYPES[nm]
        lo, hi = spec["age"]
        age[mask] = rng.integers(lo, hi + 1, size=m).astype(np.int16)
        income_mult = rng.lognormal(0.0, spec["income_sigma"], size=m)
        income[mask] = spec["income_base"] * income_mult
        traits["income_level"][mask] = income_mult
        for t in TRAITS[1:]:
            a, b = spec[t]
            arch_draw = rng.beta(a, b, size=m)
            anywhere = rng.beta(2.5, 2.5, size=m)  # population-level prior
            traits[t][mask] = (1.0 - w) * arch_draw + w * anywhere

    # Staggered onboarding over the first 60% of the horizon (everyone has
    # history before the short-eval point; credit eval filters on activity).
    horizon_days = int(cfg.months * 30.44)
    last_signup = max(30, int(horizon_days * 0.6) - 1)
    signup_day = rng.integers(0, last_signup, size=n).astype(np.int32)

    return PersonaTable(
        archetype_idx=archetype_idx.astype(np.int16),
        country=country,
        age=age,
        income_monthly=income,
        signup_day=signup_day,
        traits=traits,
    )


def redraw_as(personas: PersonaTable, idxs: np.ndarray, archetype: str, cfg: Any,
              rng: np.random.Generator) -> None:
    """Re-draw archetype, age, income, and traits for ``idxs`` in place.

    Used when mule rings must draft beyond the mule-archetype pool: drafted
    users become real mules (traits from the mule prior), so AML positives are
    persona-coherent at every scale rather than a mix of personas that merely
    happen to receive ring transfers.
    """
    m = len(idxs)
    if m == 0:
        return
    spec = ARCHETYPES[archetype]
    w = float(cfg.trait_noise)
    personas.archetype_idx[idxs] = ARCHETYPE_NAMES.index(archetype)
    lo, hi = spec["age"]
    personas.age[idxs] = rng.integers(lo, hi + 1, size=m).astype(np.int16)
    income_mult = rng.lognormal(0.0, spec["income_sigma"], size=m)
    personas.income_monthly[idxs] = spec["income_base"] * income_mult
    personas.traits["income_level"][idxs] = income_mult
    for t in TRAITS[1:]:
        a, b = spec[t]
        arch_draw = rng.beta(a, b, size=m)
        anywhere = rng.beta(2.5, 2.5, size=m)
        personas.traits[t][idxs] = (1.0 - w) * arch_draw + w * anywhere
