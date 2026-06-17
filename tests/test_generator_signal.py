"""AML signal-placement guarantees of the synthetic generator.

The AML ablation is only meaningful if the generator puts AML signal in
the right place: NOT in a single structural or field-level oracle (degree,
occupation, an exclusive txn_type) but in behavior an embedding must learn.
These pin the property that no single structural or field-level feature
separates mules, so the signal lives in relational and behavioral patterns.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.metrics import roc_auc_score

from pragmatiq import api
from pragmatiq.data.synthetic.config import WorldConfig
from pragmatiq.data.synthetic.personas import ARCHETYPE_NAMES
from pragmatiq.data.synthetic.world import World

CFG = {"n_users": 1200, "months": 14, "mule_ring_count": 12, "seed": 11,
       "eval_month_credit": 2, "eval_month_short": 8}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("sig") / "ds"
    api.synthesize(dict(CFG), out=out, write_report=False)
    aml = pq.read_table(out / "labels" / "aml.parquet").to_pandas()
    return out, dict(zip(aml["user_id"], aml["label"]))


def _user_auc(labels: dict[str, int], feature: dict[str, float]) -> float:
    uids = list(labels)
    y = np.array([labels[u] for u in uids])
    x = np.array([feature.get(u, 0.0) for u in uids])
    return float(roc_auc_score(y, x))


class TestStructuralOracleRemoved:
    def test_distinct_counterparty_auc_below_085(self, dataset) -> None:
        out, labels = dataset
        df = pq.read_table(out / "transfers.parquet").to_pandas()
        cps: dict[str, set] = {}
        for a, b in zip(df["from_user"], df["to_user"]):
            cps.setdefault(a, set()).add(b)
            cps.setdefault(b, set()).add(a)
        auc = _user_auc(labels, {u: float(len(s)) for u, s in cps.items()})
        assert auc <= 0.85, f"distinct-counterparty count is still a structural oracle: {auc:.3f}"


class TestRelationalDesign:
    def test_aml_positives_are_persona_diverse(self) -> None:
        # Mules are ordinary recruited accounts drawn from the GENERAL population,
        # NOT a distinct "mule" persona — their only signal is ring membership in
        # the transfer graph (the relational regime). A single dominant
        # archetype would let an isolated embedding read membership off persona alone.
        world = World.build(WorldConfig(**CFG))
        members = np.nonzero(world.episodes.mule_member)[0]
        assert len(members) > 30, "expected drafted rings at this scale"
        counts = np.bincount(world.personas.archetype_idx[members], minlength=len(ARCHETYPE_NAMES))
        n_distinct = int((counts > 0).sum())
        top_frac = counts.max() / len(members)
        assert n_distinct >= 3, f"AML positives span only {n_distinct} archetype(s); expected persona-diverse"
        assert top_frac <= 0.6, f"one archetype is {top_frac:.0%} of mules; mules should be persona-diverse"


class TestNoSingleFieldMarker:
    def test_occupation_does_not_separate_mules(self, dataset) -> None:
        out, labels = dataset
        prof = pq.read_table(out / "profiles.parquet").to_pandas()
        occ = {u: dict(a).get("occupation", "") for u, a in zip(prof["user_id"], prof["attributes"])}
        values = sorted(set(occ.values()))
        worst = max(
            _user_auc(labels, {u: float(occ.get(u) == v) for u in labels}) for v in values
        )
        assert worst <= 0.75, f"an occupation value separates mules at AUC {worst:.3f}"

    def test_crypto_topup_presence_not_an_oracle(self, dataset) -> None:
        out, labels = dataset
        ev = pq.read_table(out / "events.parquet", columns=["user_id", "fields"]).to_pandas()
        has_cx: dict[str, float] = {}
        for u, f in zip(ev["user_id"], ev["fields"]):
            if dict(f).get("txn_type") == "crypto_topup":
                has_cx[u] = 1.0
        auc = _user_auc(labels, has_cx)
        assert auc <= 0.95, f"crypto_topup presence alone separates mules at AUC {auc:.3f}"

    def test_atm_merchant_not_an_aml_oracle(self, dataset) -> None:
        # mule cash-out ATM merchant names are drawn from the same noised pool
        # legit withdrawals use, so the merchant string is not a single-field tell
        out, labels = dataset
        ev = pq.read_table(out / "events.parquet", columns=["user_id", "fields"]).to_pandas()
        bare: dict[str, float] = {}
        for u, f in zip(ev["user_id"], ev["fields"]):
            if dict(f).get("merchant") == "EURONET ATM":  # a fixed ATM merchant string
                bare[u] = 1.0
        auc = _user_auc(labels, bare)
        assert auc <= 0.95, f"a bare ATM merchant string separates mules at AUC {auc:.3f}"


def _event_max_single_value_auc(out, field: str, label_table: str = "fraud") -> float:
    """Max over the 40 commonest values of single-value (field==v) AUC vs the
    event-level label — detects a single field value that is a label oracle."""
    from collections import Counter

    fr = pq.read_table(out / "labels" / f"{label_table}.parquet").to_pandas()
    pos = set(zip(fr[fr["label"] == 1]["user_id"], fr[fr["label"] == 1]["ts"]))
    ev = pq.read_table(out / "events.parquet", columns=["user_id", "ts", "source", "fields"]).to_pandas()
    ev = ev[ev["source"] == "transaction"].reset_index(drop=True)
    vals = np.array([dict(f).get(field, "") for f in ev["fields"]], dtype=object)
    y = np.array([1 if (u, t) in pos else 0 for u, t in zip(ev["user_id"], ev["ts"])])
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    best = 0.5
    for v, _ in Counter(vals.tolist()).most_common(40):
        a = roc_auc_score(y, (vals == v).astype(int))
        best = max(best, a, 1.0 - a)
    return best


class TestFraudNoSingleFieldOracle:
    """Account-takeover fraud must not be readable off one field value: the
    attacker's country, OS, and verify-merchant fields each blend into the
    legitimate distribution rather than acting as a single-field tell."""

    def test_country_not_a_fraud_oracle(self, dataset) -> None:
        out, _ = dataset
        auc = _event_max_single_value_auc(out, "country")
        assert auc <= 0.85, f"a single country value separates fraud at AUC {auc:.3f}"

    def test_merchant_not_a_fraud_oracle(self, dataset) -> None:
        out, _ = dataset
        auc = _event_max_single_value_auc(out, "merchant")
        assert auc <= 0.85, f"a single merchant value separates fraud at AUC {auc:.3f}"
