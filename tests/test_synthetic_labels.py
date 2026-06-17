"""LabelOracle and calibration tests: leakage rules and moment matching."""

from __future__ import annotations

import numpy as np
import pytest

from pragmatiq.data.synthetic.calibrate import calibrate_config
from pragmatiq.data.synthetic.config import WorldConfig
from pragmatiq.data.synthetic.labels import LabelOracle
from pragmatiq.data.synthetic.simulator import UserSimulator
from pragmatiq.data.synthetic.world import World, user_rng

CFG = WorldConfig(
    n_users=200, months=16, n_merchants=800, mule_ring_count=1, seed=5,
    eval_month_credit=4, eval_month_short=9, label_noise=0.0,
)


@pytest.fixture(scope="module")
def world() -> World:
    return World.build(CFG)


def _trace(world: World, u: int):
    sim = UserSimulator(world)
    rng = user_rng(CFG.seed, u)
    return sim.run(u, rng), rng


class TestOracle:
    def test_default_window_excludes_pre_eval_insolvency(self, world: World) -> None:
        oracle = LabelOracle(world)
        tr, rng = _trace(world, 0)
        # Force an insolvency before eval: user must not be scored at all.
        tr.insolvency_day = oracle.eval_credit_day - 10
        rows = oracle.label_user(tr, rng)
        assert rows.default_12m == []

    def test_default_positive_only_within_12m_window(self, world: World) -> None:
        oracle = LabelOracle(world)
        tr, rng = _trace(world, 1)
        if not len(tr.ts):
            pytest.skip("inactive user")
        tr.insolvency_day = oracle.eval_credit_day + 30
        rows = oracle.label_user(tr, rng)
        if rows.default_12m:
            assert rows.default_12m[0][2] == 1
        tr2, rng2 = _trace(world, 1)
        tr2.insolvency_day = oracle.eval_credit_day + 400  # outside 12m
        rows2 = oracle.label_user(tr2, rng2)
        if rows2.default_12m:
            assert rows2.default_12m[0][2] == 0

    def test_churn_label_is_future_churn(self, world: World) -> None:
        """Spec semantics: label = churns within 6m AFTER eval; eligible = active at eval."""
        oracle = LabelOracle(world)
        found = pos = 0
        m = CFG.eval_month_short
        for u in range(120):
            tr, rng = _trace(world, u)
            rows = oracle.label_user(tr, rng)
            if not rows.churn_6m:
                # excluded users must be either never-active or already churned at eval
                active_at_eval = tr.churn_month == -1 or tr.churn_month > m
                signed_up = len(tr.ts) > 0 and int((tr.ts < oracle.eval_short_us).sum()) >= 1
                assert not (signed_up and active_at_eval)
                continue
            _, _, y = rows.churn_6m[0]
            churned_in_window = tr.churn_month != -1 and m < tr.churn_month <= m + 6
            assert y == int(churned_in_window)
            # eligible users were active at eval (not already churned)
            assert tr.churn_month == -1 or tr.churn_month > m
            found += 1
            pos += y
        assert found > 10
        assert pos >= 1, "expected at least one future-churn positive in 120 users"

    def test_presignup_users_can_still_churn(self, world: World) -> None:
        """Late-signup users keep their real churn month rather than the pre-signup sentinel."""
        from pragmatiq.data.synthetic.simulator import CHURNED
        seen_late_churner = False
        for u in range(world.cfg.n_users):
            signup_m = int(world.calendar.month_of_day(np.array([world.personas.signup_day[u]]))[0])
            if signup_m == 0:
                continue
            tr, _ = _trace(world, u)
            if (tr.lifecycle == CHURNED).any():
                first_churn = int(np.argmax(tr.lifecycle == CHURNED))
                assert tr.churn_month == first_churn  # real churn month, not the -1 sentinel
                assert first_churn >= signup_m
                seen_late_churner = True
        assert seen_late_churner, "no late-signup churner in sample to validate"

    def test_aml_matches_ring_membership(self, world: World) -> None:
        oracle = LabelOracle(world)
        for u in range(60):
            tr, rng = _trace(world, u)
            rows = oracle.label_user(tr, rng)
            assert rows.aml[0][2] == int(world.episodes.mule_member[u])

    def test_uplift_outcomes_monotone(self, world: World) -> None:
        oracle = LabelOracle(world)
        for u in range(40):
            tr, rng = _trace(world, u)
            for _, _, _, _, y0, y1 in oracle.label_user(tr, rng).comm_uplift:
                assert y1 >= y0


class TestCalibrate:
    def test_direct_moment_mapping(self) -> None:
        stats = {
            "archetype_shares": {"salaried": 3, "student": 1},
            "fraud_base_rate": 0.01,
            "default_rate": 0.05,
            "mcc_mix": {"grocery": 0.5, "fuel": 0.5},
            "mean_amount_by_mcc": {"grocery": 40.0},
            "n_users": 500,
        }
        out = calibrate_config(stats, base={"months": 16, "eval_month_credit": 4,
                                            "eval_month_short": 9})
        assert out["archetype_mix"]["salaried"] == pytest.approx(0.75)
        assert out["fraud_rate"] == 0.01
        assert out["default_rate"] == 0.05
        assert out["mcc_weights"] == {"grocery": 0.5, "fuel": 0.5}
        assert out["mcc_amount_mean"] == {"grocery": 40.0}
        # produces a valid WorldConfig and a world
        World.build(WorldConfig.from_dict(out))

    def test_unknown_mcc_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown MCC"):
            calibrate_config({"mcc_mix": {"nonsense": 1.0}})

    def test_activity_scale_fitting(self) -> None:
        stats = {"events_per_user_month": 12.0}
        out = calibrate_config(stats, base={"months": 14, "n_users": 200,
                                            "eval_month_credit": 2, "eval_month_short": 8,
                                            "n_merchants": 500})
        assert out["activity_scale"] != 1.0  # it moved toward the target

    def test_amount_mean_actually_applies(self) -> None:
        cfg = WorldConfig(n_users=50, months=14, n_merchants=500, seed=1,
                          eval_month_credit=2, eval_month_short=8,
                          mcc_amount_mean={"grocery": 100.0})
        w = World.build(cfg)
        from pragmatiq.data.synthetic.world import MCC_IDX
        i = MCC_IDX["grocery"]
        implied_mean = float(np.exp(w.merchants.mcc_mu[i] + 0.5 * w.merchants.mcc_sigma[i] ** 2))
        assert implied_mean == pytest.approx(100.0, rel=0.01)
