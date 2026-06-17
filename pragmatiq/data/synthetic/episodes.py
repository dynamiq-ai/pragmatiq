"""Episode injection (fraud, stress, mule) and the per-user LatentLog.

Phase A (world build) decides *who* gets episodes (``EpisodeAssignment``);
``EpisodeInjector`` realizes them as concrete events inside a user's history
during phase B. Every realized episode is recorded in the user's ``LatentLog``
— the ground truth the ``LabelOracle`` reads. Labels are consequences of these
latents via simulated behavior, never direct copies of traits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .world import DAY_US, MCC_IDX, MCC_KEYS, World

HOUR_US = 3_600_000_000
MIN_US = 60_000_000

_SCREENS_ATO = np.array(["home", "cards", "payments", "transfers"], dtype=object)
# Attacker fingerprints overlap the legit population so no single value (a far
# country, an older OS, an "INTL VERIFY" string) is a deterministic event-level
# fraud marker on its own. The fraud signature lives in the sequence (new device
# + geo jump vs the user's own home + small test txn + rapid drains + night
# timing), not in standalone token values. OS values are drawn from the same
# pool as the legit _OSES set in simulator.py.
_ATTACKER_OSES = np.array(["android_14", "android_15", "ios_17", "ios_18"], dtype=object)
_ATTACKER_CCYS = np.array(["USD", "EUR", "GBP"], dtype=object)
# Shared with the organic crypto top-ups in the simulator: crypto merchants
# must appear in normal books too, or merchant strings become an AML marker.
_CRYPTO_EXCHANGES = np.array(["BINANCE", "COINBASE", "KRAKEN.COM", "CRYPTO.COM"], dtype=object)


@dataclass
class SourceBuffer:
    """Columnar event buffer for one source: parallel arrays per field key.

    Appends may use different key sets (e.g. only p2p rows carry
    ``counterparty``); ``concat`` aligns blocks and fills missing keys with
    ``""`` (= absent in the fields map).
    """

    blocks: list[tuple[np.ndarray, dict[str, np.ndarray]]] = field(default_factory=list)

    def append(self, ts_us: np.ndarray, **columns: np.ndarray) -> None:
        """Append a block of events given as equal-length column arrays."""
        n = len(ts_us)
        if n == 0:
            return
        cols = {}
        for k, v in columns.items():
            arr = np.asarray(v, dtype=object)
            assert len(arr) == n, f"ragged column {k!r}: {len(arr)} != {n}"
            cols[k] = arr
        self.blocks.append((np.asarray(ts_us, dtype=np.int64), cols))

    def groups(self) -> list[tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]]:
        """Blocks grouped by key signature: list of (keys, ts, cols).

        Within a group every event has exactly the signature's keys — the
        property the vectorized Arrow map assembly relies on.
        """
        bysig: dict[tuple[str, ...], list[tuple[np.ndarray, dict[str, np.ndarray]]]] = {}
        order: list[tuple[str, ...]] = []
        for ts, cols in self.blocks:
            sig = tuple(cols.keys())
            if sig not in bysig:
                bysig[sig] = []
                order.append(sig)
            bysig[sig].append((ts, cols))
        out = []
        for sig in order:
            blocks = bysig[sig]
            ts = np.concatenate([b[0] for b in blocks])
            cols = {k: np.concatenate([b[1][k] for b in blocks]) for k in sig}
            out.append((sig, ts, cols))
        return out


@dataclass
class FraudEpisode:
    """Account takeover: new device + geo jump + test txn + rapid drains."""

    start_day: int
    len_days: int
    n_drains: int = 0
    total_stolen: float = 0.0


@dataclass
class StressArc:
    """2–6 months of shrinking salary, rising gambling and night activity."""

    start_month: int
    len_months: int
    severity: float
    insolvency_day: int = -1  # filled after balance simulation


@dataclass
class MuleEpisode:
    """Mule-ring window: fan-in of small credits, rapid fan-out + ATM."""

    window_start_day: int
    window_end_day: int


@dataclass
class LatentLog:
    """Ground-truth record of everything hidden that shaped a user's history."""

    user_idx: int
    fraud: FraudEpisode | None = None
    stress: StressArc | None = None
    mule: MuleEpisode | None = None

    def episodes(self) -> list[object]:
        """All realized episodes, in no particular order."""
        return [e for e in (self.fraud, self.stress, self.mule) if e is not None]


class EpisodeInjector:
    """Realizes assigned episodes as events inside one user's buffers."""

    def __init__(self, world: World) -> None:
        self.world = world

    def stress_profile(self, user_idx: int) -> tuple[np.ndarray, StressArc | None]:
        """Monthly stress intensity in [0,1] (ramping over the arc) + arc record."""
        ep = self.world.episodes
        cfg = self.world.cfg
        stress_months = np.zeros(cfg.months, dtype=np.float64)
        if not ep.stress_user[user_idx]:
            return stress_months, None
        s0 = int(ep.stress_start_month[user_idx])
        ln = int(ep.stress_len_months[user_idx])
        s1 = min(cfg.months, s0 + ln)
        sev = float(ep.stress_severity[user_idx])
        stress_months[s0:s1] = np.linspace(0.55, 1.0, max(s1 - s0, 1)) * sev
        return stress_months, StressArc(start_month=s0, len_months=ln, severity=sev)

    def inject_fraud(
        self, rng: np.random.Generator, user_idx: int, ccy: str,
        txn: SourceBuffer, app: SourceBuffer,
        cash: list[tuple[int, float]], fraud_rows: list[int],
        mcc_code: dict[str, str],
    ) -> FraudEpisode:
        """Account-takeover episode; appends events and returns the record."""
        w = self.world
        cal = w.calendar
        ep = w.episodes
        start_day = int(ep.fraud_start_day[user_idx])
        len_days = int(ep.fraud_len_days[user_idx])
        t0 = cal.start_us() + start_day * DAY_US + 22 * HOUR_US + int(rng.uniform(0, 4) * HOUR_US)
        # Geo jump to a DIFFERENT real country in the population (not the user's
        # home). Drawing from country_mix (vs a disjoint far-country set) keeps
        # the jump anomalous in sequence context without making `country` a
        # standalone fraud oracle — every value also occurs in legit data.
        home = str(w.personas.country[user_idx])
        geo_pool = [c for c in w.cfg.country_mix if c != home] or list(w.cfg.country_mix)
        far = str(geo_pool[int(rng.integers(0, len(geo_pool)))])
        new_dev = f"dev_{rng.integers(10**9, 10**10 - 1)}"
        # 1) ATO login burst from a new device with a geo jump.
        n_nav = int(rng.integers(4, 9))
        nav_ts = t0 + np.cumsum((rng.exponential(12.0, size=n_nav) * 1_000_000).astype(np.int64))
        # Attacker fingerprint is randomized per episode — constants here would
        # hand any classifier a deterministic event-level marker (label leak).
        atk_os = str(_ATTACKER_OSES[rng.integers(0, len(_ATTACKER_OSES))])
        atk_app = f"{rng.integers(8, 11)}.{rng.integers(0, 99)}"
        app.append(
            nav_ts,
            screen=_SCREENS_ATO[rng.integers(0, len(_SCREENS_ATO), size=n_nav)],
            action=np.where(rng.random(n_nav) < 0.5, "view", "tap").astype(object),
            device_id=np.full(n_nav, new_dev, dtype=object),
            os=np.full(n_nav, atk_os, dtype=object),
            app_version=np.full(n_nav, atk_app, dtype=object),
            geo_country=np.full(n_nav, far, dtype=object),
        )
        # 2) small test transaction, then 3) rapid large drains over 1–3 days.
        test_ts = int(nav_ts[-1] + 3 * MIN_US)
        n_big = int(rng.integers(4, 13))
        big_off = np.sort((rng.random(n_big) * len_days * DAY_US * 0.9).astype(np.int64))
        big_ts = test_ts + 10 * MIN_US + big_off
        drain_mcc = [MCC_IDX[k] for k in ("electronics", "jewelry", "online_retail")]
        mcc_pick = np.array(drain_mcc)[rng.integers(0, len(drain_mcc), size=n_big)]
        merch = np.array(
            [w.merchants.names[w.merchants.sample_in_mcc(int(m), rng.random(1))[0]] for m in mcc_pick],
            dtype=object,
        )
        big_amt = np.round(rng.uniform(180, 950, size=n_big), 2)
        all_ts = np.concatenate([[test_ts], big_ts]).astype(np.int64)
        all_amt = np.concatenate([[round(float(rng.uniform(0.9, 3.2)), 2)], big_amt])
        all_mcc = np.array(
            [mcc_code["online_retail"]] + [mcc_code[MCC_KEYS[m]] for m in mcc_pick], dtype=object
        )
        # The small "test" transaction is a real (noised) merchant purchase; a
        # bare verify-string here would be a fraud-exclusive event marker.
        test_merch = str(w.merchants.names[w.merchants.sample_in_mcc(MCC_IDX["online_retail"], rng.random(1))[0]])
        all_merch = np.concatenate([[test_merch], merch]).astype(object)
        n_all = len(all_ts)
        cur = str(_ATTACKER_CCYS[rng.integers(0, len(_ATTACKER_CCYS))])
        txn.append(
            all_ts,
            amount=np.array([f"{a:.2f}" for a in all_amt], dtype=object),
            currency=np.full(n_all, cur, dtype=object),
            mcc=all_mcc,
            merchant=all_merch,
            txn_type=np.full(n_all, "card_payment", dtype=object),
            channel=np.full(n_all, "online", dtype=object),
            country=np.full(n_all, far, dtype=object),
        )
        for t, a in zip(all_ts.tolist(), all_amt.tolist()):
            cash.append((int(t), -float(a)))
            fraud_rows.append(int(t))
        return FraudEpisode(start_day=start_day, len_days=len_days,
                            n_drains=n_big, total_stolen=float(all_amt.sum()))

    def inject_mule_atm(
        self, rng: np.random.Generator, user_idx: int, ccy: str,
        txn: SourceBuffer, cash: list[tuple[int, float]], mcc_code: dict[str, str],
        app: SourceBuffer | None = None, devices: list[str] | None = None,
        os_name: str = "android_14",
    ) -> MuleEpisode | None:
        """Cash-out burst during the ring window (the P2P legs are world-scheduled).

        A mule extracts the received funds fast and in a distinctive way: a
        cluster of near-limit ATM withdrawals plus crypto-exchange top-ups,
        night-skewed and tightly grouped in time. This is a *behavioral*
        signature in the event sequence (transaction types, amounts, rapid
        cadence) that the PRAGMA-style encoder picks up — and that raw transfer-graph
        degree statistics miss. It is what lets GNN+PRAGMA beat GNN+handcrafted
        in the phase-6 ablation, not just the ring topology.
        """
        ep = self.world.episodes
        cal = self.world.calendar
        country = str(self.world.personas.country[user_idx])
        w0, w1 = int(ep.mule_window[user_idx, 0]), int(ep.mule_window[user_idx, 1])
        if w0 < 0:
            return None
        s = float(self.world.cfg.mule_behavior_strength)
        if s <= 0.0:
            # Relational-AML regime: the mule has NO distinctive individual
            # cash-out fingerprint. Its only trace is ring membership in the
            # transfer graph (fan-in → layering → fan-out, time-dispersed to
            # blend with normal P2P), so the signal is purely RELATIONAL —
            # recoverable by a graph-aware model via the ring community and
            # neighbour behaviour, not by an isolated per-user probe.
            return MuleEpisode(window_start_day=w0, window_end_day=w1)
        # ATM cash-out: near-limit withdrawals clustered over 1-2 nights;
        # volume scales with mule_behavior_strength (the embedding-visible arm
        # of the AML signal — see config.py).
        n_atm = max(2, int(round(int(rng.integers(4, 10)) * s)))
        atm_day = rng.uniform(w0, max(w0 + 1, w1), size=n_atm)
        atm_ts = cal.start_us() + (atm_day * DAY_US).astype(np.int64) + \
            (rng.uniform(20, 27.5, size=n_atm) % 24 * HOUR_US).astype(np.int64)
        atm_amt = np.round(rng.uniform(180, 450, size=n_atm), 2)
        txn.append(
            atm_ts,  # the global merge sorts by ts; sorting here would mispair ts↔amount
            amount=np.array([f"{a:.2f}" for a in atm_amt], dtype=object),
            currency=np.full(n_atm, ccy, dtype=object),
            mcc=np.full(n_atm, mcc_code["atm"], dtype=object),
            # ATM cash-out merchant names are drawn from the same noised pool as
            # legitimate withdrawals, so the merchant string is not a single-field tell.
            merchant=self.world.merchants.names[
                self.world.merchants.sample_in_mcc(MCC_IDX["atm"], rng.random(n_atm))
            ].astype(object),
            txn_type=np.full(n_atm, "atm_withdrawal", dtype=object),
            channel=np.full(n_atm, "atm", dtype=object),
            country=np.full(n_atm, country, dtype=object),
        )
        # Crypto-exchange top-ups: the other classic laundering cash-out leg.
        n_cx = max(1, int(round(int(rng.integers(2, 6)) * s)))
        cx_ts = cal.start_us() + (rng.uniform(w0, max(w0 + 1, w1), size=n_cx) * DAY_US).astype(np.int64) + \
            (rng.uniform(0, 24, size=n_cx) * HOUR_US).astype(np.int64)
        cx_amt = np.round(rng.uniform(120, 600, size=n_cx), 2)
        txn.append(
            cx_ts,  # the global merge sorts by ts; sorting here would mispair ts↔amount/merchant
            amount=np.array([f"{a:.2f}" for a in cx_amt], dtype=object),
            currency=np.full(n_cx, ccy, dtype=object),
            mcc=np.full(n_cx, mcc_code["online_retail"], dtype=object),
            merchant=_CRYPTO_EXCHANGES[rng.integers(0, len(_CRYPTO_EXCHANGES), size=n_cx)],
            txn_type=np.full(n_cx, "crypto_topup", dtype=object),
            channel=np.full(n_cx, "online", dtype=object),
            country=np.full(n_cx, country, dtype=object),
        )
        for t, a in zip(atm_ts.tolist(), atm_amt.tolist()):
            cash.append((int(t), -float(a)))
        for t, a in zip(cx_ts.tolist(), cx_amt.tolist()):
            cash.append((int(t), -float(a)))
        # Night app bursts ending on the transfers screen before each cash-out
        # cluster: the in-app shadow of moving money, visible only to the
        # sequence encoder. Uses the user's own devices (this is the account
        # owner acting, unlike an ATO).
        if app is not None and devices:
            anchor_ts = np.concatenate([atm_ts, cx_ts])
            n_nav = rng.integers(2, 6, size=len(anchor_ts))
            total = int(n_nav.sum())
            anchor_of_ev = np.repeat(anchor_ts, n_nav)
            order_in = np.concatenate([np.arange(k, 0, -1) for k in n_nav])
            gap = (rng.uniform(20, 120, size=total) * order_in * 1_000_000).astype(np.int64)
            burst_ts = anchor_of_ev - gap
            last = order_in == 1
            screen = np.where(last, "transfers",
                              _SCREENS_ATO[rng.integers(0, len(_SCREENS_ATO), size=total)]).astype(object)
            dev = np.array(devices, dtype=object)[rng.integers(0, len(devices), size=total)]
            app.append(
                burst_ts,
                screen=screen,
                action=np.where(rng.random(total) < 0.55, "view", "tap").astype(object),
                device_id=dev,
                os=np.full(total, os_name, dtype=object),
                app_version=np.full(total, f"10.{rng.integers(0, 40)}", dtype=object),
                geo_country=np.full(total, country, dtype=object),
            )
        return MuleEpisode(window_start_day=w0, window_end_day=w1)
