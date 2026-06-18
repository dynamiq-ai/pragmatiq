"""Tests for phase-A world construction (calendar, merchants, graph, episodes)."""

from __future__ import annotations

import numpy as np
import pytest

from pragmatiq.data.synthetic.config import WorldConfig
from pragmatiq.data.synthetic.personas import ARCHETYPE_NAMES, TRAITS, sample_personas
from pragmatiq.data.synthetic.world import Calendar, World


@pytest.fixture(scope="module")
def small_cfg() -> WorldConfig:
    return WorldConfig(
        n_users=300, months=18, n_merchants=2000, mule_ring_count=2, seed=42,
        eval_month_credit=6, eval_month_short=11,
    )


@pytest.fixture(scope="module")
def world(small_cfg: WorldConfig) -> World:
    return World.build(small_cfg)


class TestCalendar:
    def test_horizon(self, world: World) -> None:
        cal = world.calendar
        assert cal.months == 18
        assert cal.n_days == int(cal.month_start_day[-1])
        assert len(cal.day_of_week) == cal.n_days

    def test_weekends_match_dow(self, world: World) -> None:
        cal = world.calendar
        assert np.array_equal(cal.is_weekend, cal.day_of_week >= 5)
        # 2023-01-01 was a Sunday.
        assert cal.day_of_week[0] == 6

    def test_paydays_are_business_days(self, world: World) -> None:
        cal = world.calendar
        for m, pd_day in enumerate(cal.payday):
            assert cal.month_start_day[m] <= pd_day < cal.month_start_day[m + 1]
            assert not cal.is_weekend[pd_day]
            assert not cal.is_holiday[pd_day]

    def test_seasonality_and_inflation(self, world: World) -> None:
        cal = world.calendar
        assert cal.season_mult.min() > 0.5 and cal.season_mult.max() < 1.6
        assert cal.inflation_mult[0] == pytest.approx(1.0)
        assert cal.inflation_mult[-1] > cal.inflation_mult[0]

    def test_holidays_include_christmas(self) -> None:
        cfg = WorldConfig(n_users=10, months=14, n_merchants=100, seed=0,
                          eval_month_credit=2, eval_month_short=8)
        cal = Calendar.build(cfg)
        # 2023-12-25 is day 358 from 2023-01-01.
        assert cal.is_holiday[358]


class TestMerchants:
    def test_universe_shapes(self, world: World) -> None:
        mu = world.merchants
        assert mu.n_merchants == 2000
        assert len(mu.names) == 2000
        assert all(isinstance(n, str) and n for n in mu.names[:50])

    def test_zipf_popularity_within_mcc(self, world: World) -> None:
        mu = world.merchants
        rng = np.random.default_rng(0)
        draws = mu.sample_in_mcc(0, rng.random(20_000))
        _, counts = np.unique(draws, return_counts=True)
        counts = np.sort(counts)[::-1]
        # Top merchant should dominate the tail by a wide margin (Zipf).
        assert counts[0] > 8 * counts[min(len(counts) - 1, 50)]

    def test_display_name_noise_exists(self, world: World) -> None:
        names = [str(n) for n in world.merchants.names]
        assert any("*" in n for n in names), "aggregator prefixes missing"
        assert any(any(c.isdigit() for c in n) for n in names), "store numbers missing"


class TestPersonas:
    def test_traits_in_range(self, small_cfg: WorldConfig) -> None:
        pt = sample_personas(small_cfg, np.random.default_rng(1))
        for t in TRAITS[1:]:
            assert pt.traits[t].min() >= 0 and pt.traits[t].max() <= 1
        assert pt.income_monthly.min() > 0
        assert set(np.unique(pt.archetype_idx)) <= set(range(len(ARCHETYPE_NAMES)))

    def test_archetype_mix_respected(self, small_cfg: WorldConfig) -> None:
        cfg = WorldConfig(**{**small_cfg.to_dict(), "n_users": 5000})
        pt = sample_personas(cfg, np.random.default_rng(1))
        share = (pt.archetype_idx == ARCHETYPE_NAMES.index("salaried")).mean()
        assert 0.3 < share < 0.46


class TestTransferGraph:
    def test_schedule_sorted_and_valid(self, world: World) -> None:
        tg = world.transfers
        assert np.all(np.diff(tg.ts_us) >= 0)
        assert tg.from_idx.min() >= 0 and tg.to_idx.max() < 300
        assert np.all(tg.from_idx != tg.to_idx) or len(tg.from_idx) == 0
        assert tg.amount.min() > 0

    def test_mule_rings_are_multi_hop_chains(self, world: World) -> None:
        tg = world.transfers
        assert len(tg.rings) == 2
        fan_in = tg.is_mule_leg == 1  # senders -> collectors
        cash_out = tg.is_mule_leg == 2  # distributors -> external
        layering = tg.is_mule_leg == 3  # layer L -> layer L+1
        assert fan_in.sum() > 0 and cash_out.sum() > 0 and layering.sum() > 0
        ring = tg.rings[0]
        # The chain spans >= 3 layers, so the discriminative motif is multi-hop.
        assert int(ring.layer_of_member.max()) + 1 >= 3
        members = set(int(m) for m in ring.members)
        collectors = set(int(m) for m in ring.members[ring.layer_of_member == 0])
        distributors = set(int(m) for m in
                           ring.members[ring.layer_of_member == ring.layer_of_member.max()])
        # Restrict to legs touching ring[0]'s members (legs aggregate all rings).
        # Fan-in lands only on collectors; cash-out leaves only from distributors;
        # layering edges connect ring members to ring members (the internal chain).
        fi_dst = set(int(t) for t in tg.to_idx[fan_in]) & members
        assert fi_dst <= collectors
        co_src = set(int(f) for f in tg.from_idx[cash_out]) & members
        assert co_src <= distributors
        lay_src = set(int(f) for f in tg.from_idx[layering]) & members
        lay_dst = set(int(t) for t in tg.to_idx[layering]) & members
        assert lay_src <= members and lay_dst <= members and len(lay_src) > 0

    def test_ring_legs_clamped_to_horizon(self) -> None:
        from pragmatiq.data.synthetic.world import DAY_US

        # Many rings in a short book push some windows near the end, where a
        # 30-day window plus a multi-week fan-out would otherwise schedule a
        # transfer past the last simulated day. Every leg must stay in-horizon.
        cfg = WorldConfig(n_users=400, months=14, n_merchants=800, mule_ring_count=16,
                          seed=7, eval_month_credit=2, eval_month_short=8)
        w = World.build(cfg)
        cal = w.calendar
        horizon = cal.start_us() + cal.n_days * DAY_US
        tg = w.transfers
        assert tg.ts_us.size > 0
        assert int(tg.ts_us.max()) < horizon

    def test_user_csr_views(self, world: World) -> None:
        tg = world.transfers
        u = int(tg.from_idx[0])
        out_rows = tg.user_outgoing(u)
        assert np.all(tg.from_idx[out_rows] == u)


class TestEpisodes:
    def test_assignment_rates(self, world: World) -> None:
        ep = world.episodes
        n = world.cfg.n_users
        assert ep.fraud_user.sum() == max(1, round(world.cfg.fraud_rate * n))
        assert 0 < ep.stress_user.sum() < n * 0.3
        assert ep.mule_member.sum() >= 2  # at least one member per ring

    def test_world_deterministic(self, small_cfg: WorldConfig) -> None:
        w1 = World.build(small_cfg)
        w2 = World.build(small_cfg)
        assert np.array_equal(w1.transfers.ts_us, w2.transfers.ts_us)
        assert list(w1.merchants.names[:100]) == list(w2.merchants.names[:100])
        assert np.array_equal(w1.episodes.stress_start_month, w2.episodes.stress_start_month)
