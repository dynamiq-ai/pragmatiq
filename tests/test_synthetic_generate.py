"""End-to-end generator tests: determinism, schema conformance, label sanity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from pragmatiq.core.schema import EVENTS_SCHEMA, LABEL_TASKS, PROFILES_SCHEMA, TRANSFERS_SCHEMA, label_schema
from pragmatiq.data.synthetic import WorldConfig, generate

CFG_KW = dict(
    n_users=160, months=16, n_merchants=1200, mule_ring_count=1, seed=123,
    eval_month_credit=4, eval_month_short=9, campaigns_per_month=1.5,
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _deterministic_artifacts(root: Path) -> list[Path]:
    labels = sorted((root / "labels").glob("*.parquet"))
    rels = [
        root / "events.parquet",
        root / "profiles.parquet",
        root / "transfers.parquet",
        root / "manifest.json",
        root / "realism_report.html",
        root / "realism_report.json",
        *labels,
    ]
    return [p for p in rels if p.exists()]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synth")
    generate(WorldConfig(**CFG_KW), out, n_workers=0, write_report=True)
    return out


class TestDeterminism:
    def test_same_seed_byte_identical(self, dataset: Path, tmp_path: Path) -> None:
        out2 = tmp_path / "again"
        generate(WorldConfig(**CFG_KW), out2, n_workers=0, write_report=True)
        for path in _deterministic_artifacts(dataset):
            rel = path.relative_to(dataset)
            assert _sha(path) == _sha(out2 / rel), f"{rel} differs across runs"

    def test_worker_count_invariant(self, dataset: Path, tmp_path: Path) -> None:
        out2 = tmp_path / "workers2"
        generate(WorldConfig(**CFG_KW), out2, n_workers=2, write_report=True)
        for path in _deterministic_artifacts(dataset):
            rel = path.relative_to(dataset)
            assert _sha(path) == _sha(out2 / rel), f"{rel} differs across worker counts"

    def test_different_seed_differs(self, dataset: Path, tmp_path: Path) -> None:
        out2 = tmp_path / "seed2"
        generate(WorldConfig(**{**CFG_KW, "seed": 124}), out2, n_workers=0, write_report=False)
        assert _sha(dataset / "events.parquet") != _sha(out2 / "events.parquet")

    def test_spawn_start_method_byte_identical(self, dataset: Path, tmp_path: Path, monkeypatch) -> None:
        # CI normally only exercises the fork pool; force spawn (the start method
        # on Windows/forkserver) to guard the to_dict/from_dict world rebuild path.
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
        out2 = tmp_path / "spawn"
        generate(WorldConfig(**CFG_KW), out2, n_workers=2, write_report=True)
        for path in _deterministic_artifacts(dataset):
            rel = path.relative_to(dataset)
            assert _sha(path) == _sha(out2 / rel), f"{rel} differs under spawn"


class TestMissingFieldRate:
    """missing_field_rate simulates sparse real-world feeds, deterministically."""

    @pytest.fixture(scope="class")
    def dirty(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("dirty")
        generate(WorldConfig(**{**CFG_KW, "missing_field_rate": 0.3}), out, n_workers=0,
                 write_report=False)
        return out

    @staticmethod
    def _n_field_pairs(d: Path) -> int:
        ca = pq.read_table(d / "events.parquet").column("fields").combine_chunks()
        return int(ca.offsets[-1].as_py())

    def test_zero_rate_is_byte_identical(self, dataset: Path, tmp_path: Path) -> None:
        out2 = tmp_path / "explicit0"
        generate(WorldConfig(**{**CFG_KW, "missing_field_rate": 0.0}), out2, n_workers=0,
                 write_report=False)
        assert _sha(dataset / "events.parquet") == _sha(out2 / "events.parquet")

    def test_worker_count_invariant_with_missing(self, dirty: Path, tmp_path: Path) -> None:
        out2 = tmp_path / "dirty_w2"
        generate(WorldConfig(**{**CFG_KW, "missing_field_rate": 0.3}), out2, n_workers=2,
                 write_report=False)
        assert _sha(dirty / "events.parquet") == _sha(out2 / "events.parquet")

    def test_drops_about_the_configured_fraction(self, dataset: Path, dirty: Path) -> None:
        full, dropped = self._n_field_pairs(dataset), self._n_field_pairs(dirty)
        assert dropped < full
        assert 0.6 < dropped / full < 0.8  # ~30% omitted, with sampling slack

    def test_dirty_data_tokenizes(self, dirty: Path) -> None:
        from pragmatiq.data.tokenizer import (
            PragmaTokenizer,
            TokenizerConfig,
            iter_user_records,
        )

        tok = PragmaTokenizer(
            TokenizerConfig(target_vocab=2500, n_buckets=16, categorical_threshold=150)
        ).fit(dirty)
        out = tok.encode(next(iter_user_records(dirty)))  # absent fields handled, no crash
        assert out.key_ids.size > 0


class TestSchema:
    def test_events_schema_and_order(self, dataset: Path) -> None:
        t = pq.read_table(dataset / "events.parquet")
        assert t.schema.equals(EVENTS_SCHEMA)
        df = t.to_pandas()
        assert df["user_id"].notna().all()
        assert set(df["source"].unique()) <= {"transaction", "app", "trading", "communication"}
        # events sorted by time within each user
        for _, grp in df.groupby("user_id", sort=False):
            ts = grp["ts"].astype("int64").to_numpy()
            assert np.all(np.diff(ts) >= 0)

    def test_fields_map_contents(self, dataset: Path) -> None:
        df = pq.read_table(dataset / "events.parquet").to_pandas()
        txn = df[df["source"] == "transaction"].iloc[0]
        fields = dict(txn["fields"])
        for key in ("amount", "currency", "mcc", "merchant", "txn_type", "channel"):
            assert key in fields, f"transaction missing field {key}"
        float(fields["amount"])  # parses

    def test_profiles_schema(self, dataset: Path) -> None:
        t = pq.read_table(dataset / "profiles.parquet")
        assert t.schema.equals(PROFILES_SCHEMA)
        assert len(t) == CFG_KW["n_users"]
        row = t.to_pandas().iloc[0]
        assert dict(row["attributes"]).get("country")
        keys = [e["key"] for e in row["lifelong"]]
        assert "account_opened" in keys

    def test_transfers_schema(self, dataset: Path) -> None:
        t = pq.read_table(dataset / "transfers.parquet")
        assert t.schema.equals(TRANSFERS_SCHEMA)
        assert len(t) > 0

    def test_all_label_tables_exist(self, dataset: Path) -> None:
        for task in LABEL_TASKS:
            t = pq.read_table(dataset / "labels" / f"{task}.parquet")
            assert t.num_columns >= 2, task
            assert t.schema.equals(label_schema(task)), task

    def test_report_and_manifest(self, dataset: Path) -> None:
        import json

        html = (dataset / "realism_report.html").read_text()
        assert "Hour of day" in html and "base64" in html
        metrics = json.loads((dataset / "realism_report.json").read_text())
        assert metrics["checks"]["events_per_user_long_tail"]["pass"] is True
        assert metrics["checks"]["hour_day_night_structure"]["pass"] is True
        assert set(metrics["checks"]) == {
            "events_per_user_long_tail",
            "hour_day_night_structure",
            "merchant_zipf_concentration",
            "amounts_differ_by_mcc",
            "calibration_default_rate_residual",
            "calibration_fraud_user_rate_residual",
        }
        assert "label_prevalence" in metrics
        assert "calibration_residuals" in metrics
        for key in ("default_rate", "fraud_user_rate"):
            assert set(metrics["calibration_residuals"][key]) == {
                "actual", "target", "residual", "abs_residual"
            }
        assert (dataset / "manifest.json").exists()


class TestLabelSanity:
    def test_fraud_rows_match_events(self, dataset: Path) -> None:
        fraud = pq.read_table(dataset / "labels" / "fraud.parquet").to_pandas()
        if fraud.empty:
            pytest.skip("no fraud episode hit in tiny sample")
        ev = pq.read_table(dataset / "events.parquet").to_pandas()
        ev_idx = set(zip(ev["user_id"], ev["ts"].astype("int64")))
        pairs = zip(fraud["user_id"], fraud["ts"].astype("int64"))
        for uid, ts in pairs:
            assert (uid, int(ts)) in ev_idx

    def test_recurring_rows_match_events(self, dataset: Path) -> None:
        rec = pq.read_table(dataset / "labels" / "recurring.parquet").to_pandas()
        assert not rec.empty
        ev = pq.read_table(dataset / "events.parquet").to_pandas()
        ev_idx = set(zip(ev["user_id"], ev["ts"].astype("int64")))
        sample = rec.sample(min(len(rec), 200), random_state=0)
        pairs = zip(sample["user_id"], sample["ts"].astype("int64"))
        for uid, ts in pairs:
            assert (uid, int(ts)) in ev_idx

    def test_comm_uplift_potential_outcomes(self, dataset: Path) -> None:
        cu = pq.read_table(dataset / "labels" / "comm_uplift.parquet").to_pandas()
        assert not cu.empty
        assert set(cu["treated"].unique()) <= {0, 1}
        # monotone potential outcomes by construction: y1 >= y0
        assert (cu["y1"] >= cu["y0"]).all()

    def test_aml_covers_all_users_and_matches_rings(self, dataset: Path) -> None:
        aml = pq.read_table(dataset / "labels" / "aml.parquet").to_pandas()
        assert len(aml) == CFG_KW["n_users"]
        assert list(aml.columns) == ["user_id", "observed_through", "label"]
        assert aml["label"].sum() >= 2

    def test_no_label_leakage_in_time(self, dataset: Path) -> None:
        """Forecast labels carry eval_ts; fraud/recurring rows reference real events."""
        for task in ("default_12m", "churn_6m", "ltv_positive"):
            t = pq.read_table(dataset / "labels" / f"{task}.parquet").to_pandas()
            if t.empty:
                continue
            assert t["eval_ts"].nunique() == 1  # single global eval point per task


class TestStochasticProcesses:
    def test_nhpp_hour_of_day_structure(self, dataset: Path) -> None:
        """The non-homogeneous Poisson spending process modulates intensity by the
        hour-of-day curve, so generated events show clear day/night structure."""
        import numpy as np

        ev = pq.read_table(dataset / "events.parquet", columns=["ts"]).to_pandas()
        hours = ((ev["ts"].astype("int64") // 3_600_000_000) % 24).to_numpy()
        counts = np.bincount(hours, minlength=24)
        day = counts[[10, 12, 13, 17, 18]].mean()    # peak hours (HOUR_CURVE)
        night = counts[[1, 2, 3, 4]].mean()           # trough hours
        assert day > 3 * night, f"no day/night NHPP structure: day~{day:.0f} night~{night:.0f}"


class TestRealismV2:
    """Generator v2 realism: FX amounts, payday schedules, market hours, fee feedback."""

    @staticmethod
    def _txn(dataset: Path):
        ev = pq.read_table(dataset / "events.parquet").to_pandas()
        ev = ev[ev["source"] == "transaction"].copy()
        f = ev["fields"].map(dict)
        for k in ("amount", "currency", "mcc", "txn_type", "country", "merchant", "channel"):
            ev[k] = f.map(lambda d, k=k: d.get(k))
        ev["amount"] = ev["amount"].astype(float)
        return ev

    def test_amounts_are_in_the_transaction_currency(self, dataset: Path) -> None:
        from pragmatiq.data.synthetic.config import FX_PER_GBP

        ev = self._txn(dataset)
        card = ev[(ev["txn_type"] == "card_payment") & (ev["mcc"] == "5411")]  # grocery
        med = card.groupby("currency")["amount"].median()
        assert "GBP" in med and "EUR" in med
        assert med["EUR"] / med["GBP"] == pytest.approx(FX_PER_GBP["EUR"], rel=0.25)
        if "PLN" in med:
            assert med["PLN"] / med["GBP"] == pytest.approx(FX_PER_GBP["PLN"], rel=0.35)
        fee = ev[ev["txn_type"] == "overdraft_fee"]
        assert set(fee[fee["currency"] == "GBP"]["amount"].round(2)) <= {5.0}
        assert set(fee[fee["currency"] == "EUR"]["amount"].round(2)) <= {round(5 * FX_PER_GBP["EUR"], 2)}
        # A legitimate event's currency follows its country, never the user's home
        # currency alone (account-takeover drains carry the attacker's card currency).
        from pragmatiq.data.synthetic.config import COUNTRY_CCY

        fraud = pq.read_table(dataset / "labels" / "fraud.parquet").to_pandas()
        stolen = set(zip(fraud["user_id"], fraud["ts"].astype("int64")))
        legit = ~np.array([(u, t) in stolen for u, t in zip(ev["user_id"], ev["ts"].astype("int64"))])
        mism = ev[legit & (ev["txn_type"] == "card_payment").to_numpy()
                  & (ev["country"].map(COUNTRY_CCY) != ev["currency"]).to_numpy()]
        assert len(mism) == 0

    def test_salary_paydays_vary_across_users(self, dataset: Path) -> None:
        ev = self._txn(dataset)
        sal = ev[ev["merchant"] == "EMPLOYER PAYROLL"].copy()
        dom = ((sal["ts"].astype("int64") // 86_400_000_000).map(
            lambda d: (np.datetime64(0, "D") + int(d)).astype("datetime64[D]").item().day))
        sal["dom"] = dom
        per_user = sal.groupby("user_id")["dom"].agg(lambda s: s.median())
        assert per_user.between(22, 25).any()  # fixed-date / four-weekly payers
        assert (per_user >= 28).any()  # last-business-day payers
        # Four-weekly payers see 13 credits a year: some users have >1 credit in a month.
        sal["month"] = (sal["ts"].astype("int64") // 86_400_000_000 // 30)
        assert (sal.groupby(["user_id", "month"]).size() > 1).any()

    def test_equity_orders_keep_market_hours_crypto_does_not(self, dataset: Path) -> None:
        ev = pq.read_table(dataset / "events.parquet").to_pandas()
        tr = ev[ev["source"] == "trading"].copy()
        assert len(tr) > 50
        tr["instrument"] = tr["fields"].map(lambda d: dict(d)["instrument"])
        ts = tr["ts"].astype("int64")
        tr["hour"] = (ts // 3_600_000_000) % 24
        tr["dow"] = ((ts // 86_400_000_000) + 3) % 7
        crypto = tr["instrument"].isin(["BTC", "ETH", "SOL"])
        eq = tr[~crypto]
        assert (eq["dow"] < 5).all()  # no weekend equity orders
        assert (eq["hour"].between(7, 21)).mean() > 0.97  # market hours (+ pre-market)
        assert (eq["hour"] < 6).mean() < 0.01  # essentially nothing overnight
        if crypto.sum() > 30:
            assert (tr[crypto]["dow"] >= 5).any()

    def test_overdraft_fees_debit_the_balance(self) -> None:
        from pragmatiq.data.synthetic.episodes import SourceBuffer
        from pragmatiq.data.synthetic.simulator import UserSimulator
        from pragmatiq.data.synthetic.world import World

        cfg = WorldConfig(n_users=20, months=14, n_merchants=200, seed=3,
                          eval_month_credit=2, eval_month_short=8)
        sim = UserSimulator(World.build(cfg))
        cal = sim.world.calendar
        t0 = cal.start_us()
        cash = [(t0 + 86_400_000_000 * d, -100.0) for d in range(1, 40)]  # steadily overdrawn
        txn = SourceBuffer()
        _, min_bal, end_bal = sim._balance(np.random.default_rng(0), 1000.0, txn, "GBP", list(cash), 0)
        n_fee = sum(len(ts) for _, ts, _ in txn.groups())
        assert n_fee > 0
        opening = 1000.0 * np.random.default_rng(0).uniform(0.15, 1.1)  # _balance's first draw
        assert end_bal == pytest.approx(opening + sum(a for _, a in cash) - 5.0 * n_fee)
        assert min_bal == pytest.approx(end_bal)
