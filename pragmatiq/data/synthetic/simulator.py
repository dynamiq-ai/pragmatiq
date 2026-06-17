"""Per-user behavioral simulator (phase B): one user's 25-month event history.

``UserSimulator.run(persona_row, world, rng)`` produces a ``UserTrace``:
columnar event buffers (per source), profile attributes, lifelong milestones,
recurring-series membership, episode realizations, and the post-hoc facts the
``LabelOracle`` needs (balance trajectory, lifecycle path, realized profit).

Everything is vectorized with numpy within a user; the only Python-level loops
are over months (≤ horizon) and small structures (sessions' Markov steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WorldConfig
from .episodes import _CRYPTO_EXCHANGES, EpisodeInjector, LatentLog, SourceBuffer
from .personas import ARCHETYPE_NAMES, ARCHETYPES
from .world import DAY_US, MCC_CATALOG, MCC_IDX, MCC_KEYS, World

# mcc_key -> 4-digit MCC code, from the world catalog
MCC_CATALOG_CODE: dict[str, str] = {row[1]: row[0] for row in MCC_CATALOG}

HOUR_US = 3_600_000_000
MIN_US = 60_000_000

# Lifecycle states. PRESIGNUP fills months before onboarding; it is distinct
# from CHURNED so that churn detection (first CHURNED month) is not corrupted
# for staggered signups.
ONBOARDING, ACTIVE, LOW, DORMANT, CHURNED, PRESIGNUP = 0, 1, 2, 3, 4, 5
STATE_ACTIVITY = np.array([0.55, 1.0, 0.35, 0.04, 0.0, 0.0])

_SUBSCRIPTIONS: list[tuple[str, str, float]] = [  # (merchant display, mcc_key, monthly price)
    ("SPOTIFY", "streaming", 11.99),
    ("NETFLIX.COM", "streaming", 12.99),
    ("DISNEY PLUS", "streaming", 7.99),
    ("YOUTUBE PREMIUM", "streaming", 13.99),
    ("APPLE.COM/BILL", "streaming", 2.99),
    ("AMAZON PRIME", "online_retail", 8.99),
    ("AUDIBLE UK", "streaming", 7.99),
    ("PUREGYM", "gym", 27.99),
    ("THE GYM GROUP", "gym", 24.99),
    ("ICLOUD STORAGE", "streaming", 3.49),
    ("NOW TV", "streaming", 9.99),
    ("NYTIMES DIGITAL", "books", 4.00),
    ("HELLOFRESH", "grocery", 41.99),
    ("VODAFONE", "telecom", 22.0),
    ("O2 UK", "telecom", 25.0),
    ("EE LIMITED", "telecom", 28.0),
]

_SCREENS = np.array(
    ["home", "cards", "payments", "transfers", "analytics", "trading", "crypto", "rewards", "support", "settings"],
    dtype=object,
)
# Row-stochastic screen-to-screen transition matrix (Markov walk).
_SCREEN_T = np.array(
    [
        # home  card  pay  xfer  ana  trad  cry  rew  sup  set
        [0.10, 0.18, 0.22, 0.14, 0.12, 0.06, 0.04, 0.06, 0.03, 0.05],
        [0.30, 0.10, 0.20, 0.08, 0.08, 0.02, 0.02, 0.08, 0.06, 0.06],
        [0.28, 0.12, 0.15, 0.15, 0.10, 0.02, 0.02, 0.06, 0.04, 0.06],
        [0.30, 0.08, 0.20, 0.12, 0.10, 0.02, 0.02, 0.04, 0.06, 0.06],
        [0.32, 0.10, 0.12, 0.08, 0.15, 0.06, 0.04, 0.06, 0.03, 0.04],
        [0.18, 0.04, 0.06, 0.04, 0.12, 0.35, 0.12, 0.03, 0.03, 0.03],
        [0.18, 0.04, 0.06, 0.04, 0.10, 0.18, 0.30, 0.04, 0.03, 0.03],
        [0.36, 0.10, 0.10, 0.06, 0.08, 0.04, 0.04, 0.12, 0.04, 0.06],
        [0.34, 0.08, 0.10, 0.06, 0.06, 0.02, 0.02, 0.04, 0.20, 0.08],
        [0.36, 0.10, 0.10, 0.06, 0.06, 0.02, 0.02, 0.04, 0.06, 0.18],
    ]
)
_SCREEN_T_CUM = np.cumsum(_SCREEN_T, axis=1)

_INSTRUMENTS = np.array(
    ["AAPL", "TSLA", "NVDA", "SPY", "QQQ", "VOD.L", "BARC.L", "BTC", "ETH", "SOL", "MSFT", "AMZN"],
    dtype=object,
)
_OSES = np.array(["android_14", "android_15", "ios_17", "ios_18"], dtype=object)

_CCY = {"GB": "GBP", "IE": "EUR", "FR": "EUR", "DE": "EUR", "ES": "EUR", "PL": "PLN", "LT": "EUR"}


@dataclass
class UserTrace:
    """Everything the writer and the LabelOracle need for one user.

    Events use a grouped-dense layout: ``groups[g] = (key_signature, columns)``
    where every event of group ``g`` has exactly those keys;
    ``(group_of_event[i], row_of_event[i])`` locate event ``i``'s values.
    This lets Arrow map assembly run via fancy indexing only.
    """

    user_idx: int
    user_id: str
    ts: np.ndarray  # int64 µs, time-sorted
    source: np.ndarray  # int8 index into schema.SOURCES
    group_of_event: np.ndarray  # int16
    row_of_event: np.ndarray  # int32
    groups: list[tuple[tuple[str, ...], dict[str, np.ndarray]]]
    # profile
    attributes: dict[str, str]
    lifelong: list[tuple[str, int]]
    # label facts
    recurring_rows: list[tuple[int, str]]  # (ts_us, series_id) of recurring occurrences
    fraud_rows: list[int]  # ts_us of fraudulent transactions
    insolvency_day: int  # day offset of first insolvency, -1 if none
    churn_month: int  # month the user reached CHURNED, -1 if never
    lifecycle: np.ndarray  # int8[months]
    monthly_spend: np.ndarray  # float64[months] card spend (profit input)
    monthly_trades: np.ndarray  # float64[months] traded notional
    monthly_fx: np.ndarray  # float64[months] foreign-currency spend
    is_premium: bool
    comm_rows: list[tuple[str, int, int, int, int]]  # (campaign_id, ts, treated, y0, y1)
    min_balance: float
    end_balance: float
    latent: LatentLog | None = None


def _round2str(x: np.ndarray) -> np.ndarray:
    """Format float array as 2dp strings (object dtype) — stable across runs."""
    return np.array([f"{v:.2f}" for v in np.asarray(x, dtype=np.float64)], dtype=object)


_NIGHT_HOURS = np.array([0, 1, 2, 3, 4, 22, 23])


def _nhpp_thinning(
    rng: np.random.Generator,
    day0: int,
    n_days: int,
    daily_rate: np.ndarray,
    hour_curve: np.ndarray,
    night_boost_per_day: np.ndarray | None = None,
) -> np.ndarray:
    """Non-homogeneous Poisson sampler via thinning.

    ``daily_rate[d]`` is the expected events on day ``day0+d``; ``hour_curve``
    (mean 1) modulates intensity within the day. ``night_boost_per_day`` (if
    given) scales the night hours on each day independently, so transient
    effects like a stress arc raise late-night activity only on their own days.
    Candidates are drawn from a homogeneous process at ``lam_max`` and accepted
    with prob ``λ(t)/lam_max``. Returns µs offsets from midnight of ``day0``.
    """
    if n_days <= 0 or daily_rate.sum() <= 0:
        return np.zeros(0, dtype=np.int64)
    day_hour = np.outer(daily_rate / 24.0, hour_curve)  # (n_days, 24)
    if night_boost_per_day is not None:
        day_hour[:, _NIGHT_HOURS] *= night_boost_per_day[:, None]
    lam_hour = day_hour.ravel()
    lam_max = float(lam_hour.max())
    if lam_max <= 0:
        return np.zeros(0, dtype=np.int64)
    horizon_hours = n_days * 24
    n_cand = rng.poisson(lam_max * horizon_hours)
    if n_cand == 0:
        return np.zeros(0, dtype=np.int64)
    t = rng.random(n_cand) * horizon_hours  # hours since day0 midnight
    slot = t.astype(np.int64)
    accept = rng.random(n_cand) < (lam_hour[slot] / lam_max)
    return (t[accept] * HOUR_US).astype(np.int64)


class UserSimulator:
    """Simulates one user's full history month by month (Phase 1)."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.cfg: WorldConfig = world.cfg
        # Per-transaction MCC base weights: catalog defaults, overridden where
        # phase-1b calibration provides target category shares.
        base = np.array([row[6] for row in MCC_CATALOG], dtype=np.float64)
        if world.cfg.mcc_weights:
            for k, v in world.cfg.mcc_weights.items():
                base[MCC_IDX[k]] = float(v)
        self._base_mcc_weights = base

    # ------------------------------------------------------------------ lifecycle
    def _lifecycle(self, rng: np.random.Generator, signup_month: int, churn_hazard: float,
                   stress_months: np.ndarray, archetype: str) -> np.ndarray:
        """Markov chain over months: onboarding → active ⇄ low → dormant → churned.

        Months before ``signup_month`` are PRESIGNUP (not CHURNED), so the first
        CHURNED month is always a genuine churn event.
        """
        months = self.cfg.months
        path = np.full(months, PRESIGNUP, dtype=np.int8)
        state = ONBOARDING
        dormant_pull = 0.30 if archetype == "dormant" else 0.0
        for m in range(signup_month, months):
            path[m] = state
            h = churn_hazard * (1.5 if stress_months[m] > 0 else 1.0)
            u = rng.random()
            if state == ONBOARDING:
                state = ACTIVE if u < 0.85 else LOW
            elif state == ACTIVE:
                p_low = 0.06 + 0.18 * h + dormant_pull
                state = LOW if u < p_low else ACTIVE
            elif state == LOW:
                if u < 0.30 - 0.15 * h:
                    state = ACTIVE
                elif u < 0.30 + 0.22 * h + dormant_pull:
                    state = DORMANT
            elif state == DORMANT:
                if u < 0.08:
                    state = LOW
                elif u < 0.08 + 0.25 * h + dormant_pull * 0.5:
                    state = CHURNED
            # CHURNED is absorbing
        return path

    # ------------------------------------------------------------------ main entry
    def run(self, user_idx: int, rng: np.random.Generator) -> UserTrace:
        """Simulate user ``user_idx``; all randomness comes from ``rng``."""
        w, cfg, cal = self.world, self.cfg, self.world.calendar
        p = w.personas
        arch = ARCHETYPE_NAMES[int(p.archetype_idx[user_idx])]
        spec = ARCHETYPES[arch]
        tr = {k: float(v[user_idx]) for k, v in p.traits.items()}
        country = str(p.country[user_idx])
        ccy = _CCY.get(country, "EUR")
        income = float(p.income_monthly[user_idx])
        signup_day = int(p.signup_day[user_idx])
        signup_month = int(cal.month_of_day(np.array([signup_day]))[0])
        user_id = f"u_{user_idx:08d}"

        ep = w.episodes
        injector = EpisodeInjector(w)
        latent = LatentLog(user_idx=user_idx)
        stress_months, latent.stress = injector.stress_profile(user_idx)

        lifecycle = self._lifecycle(rng, signup_month, tr["churn_hazard"], stress_months, arch)
        churned = lifecycle == CHURNED
        churn_month = int(np.argmax(churned)) if churned.any() else -1
        act_mult = STATE_ACTIVITY[lifecycle] * cfg.activity_scale

        txn = SourceBuffer()
        app = SourceBuffer()
        trd = SourceBuffer()
        com = SourceBuffer()
        recurring_rows: list[tuple[int, str]] = []
        fraud_rows: list[int] = []
        cash: list[tuple[int, float]] = []  # (ts_us, signed amount) for balance tracking

        monthly_spend = np.zeros(cfg.months)
        monthly_trades = np.zeros(cfg.months)
        monthly_fx = np.zeros(cfg.months)

        # Devices/OS are fixed per user so app sessions (standalone and the ones
        # that precede a payment) share identities.
        n_devices = 1 + int(rng.random() < 0.35 * tr["tech_savviness"] + 0.1)
        devices = [f"dev_{rng.integers(10**9, 10**10 - 1)}" for _ in range(n_devices)]
        os_name = str(_OSES[rng.integers(0, len(_OSES))])

        rent_m, subs_m = self._recurring(rng, user_idx, arch, spec, income, stress_months,
                                         act_mult, signup_month, ccy, txn, recurring_rows,
                                         cash, monthly_spend)
        self._spending(rng, user_idx, arch, spec, tr, income, rent_m + subs_m, stress_months,
                       act_mult, signup_day, country, ccy, txn, cash, monthly_spend, monthly_fx,
                       app, devices, os_name)
        self._sessions(rng, spec, tr, act_mult, signup_day, country, app, devices, os_name)
        is_premium = self._maybe_premium(rng, tr, income)
        self._trading(rng, tr, act_mult, signup_day, ccy, trd, cash, monthly_trades)
        comm_rows = self._comms(rng, user_idx, tr, lifecycle, signup_day, com, app, devices, country)
        self._p2p(user_idx, ccy, txn, cash)
        if ep.fraud_user[user_idx]:
            latent.fraud = injector.inject_fraud(rng, user_idx, ccy, txn, app, cash,
                                                 fraud_rows, MCC_CATALOG_CODE)
        if ep.mule_member[user_idx]:
            latent.mule = injector.inject_mule_atm(rng, user_idx, ccy, txn, cash, MCC_CATALOG_CODE,
                                                   app=app, devices=devices, os_name=os_name)

        insolvency_day, min_bal, end_bal = self._balance(
            rng, income, txn, ccy, cash, signup_day, country
        )
        if latent.stress is not None:
            latent.stress.insolvency_day = insolvency_day

        ts, source, group_of_event, row_of_event, groups = self._merge(txn, app, trd, com)
        attributes, lifelong = self._profile(rng, user_idx, arch, tr, income, country,
                                             signup_day, is_premium, devices)

        return UserTrace(
            user_idx=user_idx, user_id=user_id, ts=ts, source=source,
            group_of_event=group_of_event, row_of_event=row_of_event, groups=groups,
            attributes=attributes, lifelong=lifelong, recurring_rows=recurring_rows,
            fraud_rows=fraud_rows, insolvency_day=insolvency_day, churn_month=churn_month,
            lifecycle=lifecycle, monthly_spend=monthly_spend, monthly_trades=monthly_trades,
            monthly_fx=monthly_fx, is_premium=is_premium, comm_rows=comm_rows,
            min_balance=min_bal, end_balance=end_bal, latent=latent,
        )

    # ------------------------------------------------------------------ recurring
    def _recurring(self, rng, user_idx, arch, spec, income, stress_months, act_mult,
                   signup_month, ccy, txn: SourceBuffer, recurring_rows, cash,
                   monthly_spend) -> tuple[float, float]:
        """Salary/rent/subscription series; returns (rent, subs) monthly costs."""
        cal, cfg = self.world.calendar, self.cfg
        infl = cal.inflation_mult
        country = str(self.world.personas.country[user_idx])
        # Accumulate rows, then flush as one dense block (fast path for Arrow).
        rows: list[tuple[int, float, str, str, str, str]] = []  # ts, amt, merchant, mcc, type, channel

        def emit(day: int, hour_f: float, amount: float, merchant: str, mcc_key: str,
                 txn_type: str, series: str | None) -> None:
            ts = cal.start_us() + day * DAY_US + int(hour_f * HOUR_US)
            sign = 1.0 if txn_type in ("credit_transfer", "p2p_in", "refund") else -1.0
            # Bank-transfer legs (salary, rent, top-ups) are not card payments,
            # so they carry the no-MCC sentinel "0000" rather than a retail code
            # — keeps per-MCC amount distributions clean (gate-1 realism check).
            mcc_code = "0000" if mcc_key == "income" else MCC_CATALOG_CODE.get(mcc_key, "0000")
            rows.append((ts, abs(amount), merchant, mcc_code,
                         txn_type, "direct_debit" if sign < 0 else "transfer"))
            cash.append((ts, sign * abs(amount)))
            if series is not None:
                recurring_rows.append((ts, series))

        # --- salary / pension / student loan / invoices
        kind = spec["salary_kind"]
        for m in range(signup_month, cfg.months):
            if act_mult[m] <= 0:
                break
            stress = stress_months[m]
            pay_mult = max(0.0, 1.0 - 1.6 * stress)  # stress shrinks, then stops, salary
            if rng.random() < 0.02:  # occasional missed/late month even when healthy
                pay_mult *= rng.uniform(0.0, 0.6)
            if pay_mult <= 0.05:
                continue
            base = income * float(infl[cal.payday[m]])
            if kind == "monthly":
                day = int(cal.payday[m]) + int(rng.integers(-1, 2))
                day = min(max(day, int(cal.month_start_day[m])), int(cal.month_start_day[m + 1]) - 1)
                emit(day, 9 + rng.random() * 2, base * pay_mult * rng.uniform(0.98, 1.02),
                     "EMPLOYER PAYROLL", "income", "credit_transfer", f"sal_{user_idx}")
            elif kind == "pension":
                day = int(cal.month_start_day[m])
                emit(day, 8 + rng.random() * 2, base * pay_mult, "STATE PENSION", "income",
                     "credit_transfer", f"pen_{user_idx}")
            elif kind == "invoices":
                k = int(rng.integers(1, 4))
                for _ in range(k):  # k invoices summing to ~base on average
                    day = int(rng.integers(cal.month_start_day[m], cal.month_start_day[m + 1]))
                    emit(day, 10 + rng.random() * 6,
                         base * pay_mult * rng.uniform(0.85, 1.15) / k,
                         "CLIENT INVOICE", "income", "credit_transfer", f"inv_{user_idx}")
            elif kind == "loan" and m % 3 == signup_month % 3:  # termly student finance
                day = int(rng.integers(cal.month_start_day[m], cal.month_start_day[m] + 5))
                emit(day, 9 + rng.random() * 2, base * 3.0 * pay_mult, "STUDENT FINANCE", "income",
                     "credit_transfer", f"sfe_{user_idx}")
            elif kind == "none" and rng.random() < 0.7:
                # Secondary account: sporadic top-ups from the user's main bank
                # fund whatever they spend here.
                day = int(rng.integers(cal.month_start_day[m], cal.month_start_day[m + 1]))
                emit(day, 8 + rng.random() * 12, income * 0.42 * rng.uniform(0.7, 1.4),
                     "ACCOUNT TOP-UP", "income", "credit_transfer", None)

        # --- rent on the 1st
        rent = 0.0
        if rng.random() < spec["rent_prob"]:
            rent = income * rng.uniform(0.26, 0.42)
            for m in range(signup_month + 1, cfg.months):
                if act_mult[m] <= 0.05:
                    break
                day = int(cal.month_start_day[m])
                amt = rent * float(infl[day]) * rng.uniform(0.999, 1.001)
                emit(day, 7 + rng.random() * 3, amt, "LANDLORD STANDING ORDER", "income",
                     "standing_order", f"rent_{user_idx}")
                monthly_spend[m] += amt

        # --- subscriptions: start/cancel with jitter and price drift
        n_subs = int(rng.poisson(spec["subs_lambda"]))
        subs_pick = rng.choice(len(_SUBSCRIPTIONS), size=min(n_subs, len(_SUBSCRIPTIONS)), replace=False)
        subs_total = float(sum(_SUBSCRIPTIONS[int(i)][2] for i in np.atleast_1d(subs_pick)))
        for si, sub_i in enumerate(np.atleast_1d(subs_pick)):
            name, mcc_key, price = _SUBSCRIPTIONS[int(sub_i)]
            start_m = int(rng.integers(signup_month, max(signup_month + 1, cfg.months - 1)))
            dom = int(rng.integers(1, 28))
            sid = f"sub_{user_idx}_{si}"
            cancel_after = None
            for m in range(start_m, cfg.months):
                if act_mult[m] <= 0.02:
                    break
                if cancel_after is not None and m >= cancel_after:
                    break
                cancel_p = 0.02 + 0.10 * stress_months[m]
                if rng.random() < cancel_p:
                    cancel_after = m + 1
                day = min(int(cal.month_start_day[m]) + dom - 1 + int(rng.integers(-1, 2)),
                          int(cal.month_start_day[m + 1]) - 1)
                day = max(day, int(cal.month_start_day[m]))
                amt = price * float(infl[day]) * (1.0 + 0.04 * (rng.random() < 0.04))
                emit(day, rng.random() * 24, amt, name, mcc_key, "direct_debit", sid)
                monthly_spend[m] += amt

        if rows:
            n = len(rows)
            txn.append(
                np.array([r[0] for r in rows], dtype=np.int64),
                amount=np.array([f"{r[1]:.2f}" for r in rows], dtype=object),
                currency=np.full(n, ccy, dtype=object),
                mcc=np.array([r[3] for r in rows], dtype=object),
                merchant=np.array([r[2] for r in rows], dtype=object),
                txn_type=np.array([r[4] for r in rows], dtype=object),
                channel=np.array([r[5] for r in rows], dtype=object),
                country=np.full(n, country, dtype=object),
            )
        return rent, subs_total

    # ------------------------------------------------------------------ spending
    def _spending(self, rng, user_idx, arch, spec, tr, income, fixed_costs, stress_months,
                  act_mult, signup_day, country, ccy, txn: SourceBuffer, cash,
                  monthly_spend, monthly_fx, app: SourceBuffer, devices: list[str],
                  os_name: str) -> None:
        cal, w = self.world.calendar, self.world
        mu_shift = 0.35 * np.log1p(income / 2500.0)  # richer users spend more per txn

        # MCC mixture for this persona (tilts × stress-driven gambling boost).
        # ``mcc_weights`` (set by phase-1b calibration) overrides the catalog's
        # per-transaction category shares so the realized MCC mix matches the
        # target aggregate; persona tilts then perturb it.
        base_w = self._base_mcc_weights
        tilt = np.ones(len(MCC_KEYS))
        for k, v in spec["mcc_tilt"].items():
            tilt[MCC_IDX[k]] = v
        pers_w = base_w * tilt
        gambling_i = MCC_IDX["gambling"]
        # Trait-level gambling taste (mild, noisy): stressed risk-takers gamble
        # more even outside acute arcs.
        pers_w[gambling_i] *= 1.0 + 1.5 * tr["financial_stress"] * tr["risk_appetite"]

        # Personal merchant loyalty pools: a few favorites per likely MCC.
        pool: dict[int, np.ndarray] = {}
        for mi in range(len(MCC_KEYS)):
            n_fav = 3 if pers_w[mi] > 0.01 else 1
            pool[mi] = w.merchants.sample_in_mcc(mi, rng.random(n_fav))

        # Day-level intensity over the user's active life. The rate is coupled
        # to an income-derived budget so spending tracks means: without this,
        # high-propensity users drift insolvent regardless of stress arcs and
        # the credit label loses its causal anchor.
        day0 = signup_day
        n_days = cal.n_days - day0
        if n_days <= 0:
            return
        mean_amt = float(np.sum((pers_w / pers_w.sum()) * np.exp(
            w.merchants.mcc_mu + mu_shift + 0.5 * w.merchants.mcc_sigma**2)))
        disposable = max(income - fixed_costs, income * 0.12)
        budget = disposable * (0.45 + 0.40 * tr["spend_propensity"])
        if spec["salary_kind"] == "none":  # secondary account: spend ≈ top-up inflow
            budget = min(budget, income * 0.28)
        # No lower clip: low-income users must be able to spend little, or the
        # budget loses meaning and they all drift insolvent.
        rate_m = float(min(budget / max(mean_amt, 1.0), 1.9 * spec["base_txn_rate"]))
        days = np.arange(day0, cal.n_days)
        m_of_day = cal.month_of_day(days)
        daily = (
            rate_m / 30.44
            * act_mult[m_of_day]
            * cal.season_mult[days]
            * cal.DOW_MULT[cal.day_of_week[days]]
            * (1.0 + 0.15 * stress_months[m_of_day])  # desperation spending during arcs
        )
        # Payday bumps: +60% on payday and the 2 days after.
        for m in range(self.cfg.months):
            pd = int(cal.payday[m])
            sel = (days >= pd) & (days <= pd + 2)
            daily[sel] *= 1.6

        # Mule dormant-then-active arc: a quiet spell before the ring window,
        # elevated activity from the window on. A *behavioral* fingerprint the
        # sequence encoder can detect (graph degree cannot); no RNG consumed,
        # so non-mule users' streams are unchanged.
        mule_w0 = int(self.world.episodes.mule_window[user_idx, 0])
        if mule_w0 >= 0:
            s = float(self.cfg.mule_behavior_strength)
            quiet_days = 30 + mule_w0 % 16  # 30-45d, deterministic per user
            pre = (days >= mule_w0 - quiet_days) & (days < mule_w0)
            daily[pre] *= max(0.0, 1.0 - 0.7 * min(s, 1.4))
            daily[days >= mule_w0] *= 1.0 + 0.3 * s

        # Late-night activity rises only DURING the arc (not the whole history):
        # a per-day night multiplier confined to the stressed months.
        night_per_day = np.ones(n_days, dtype=np.float64)
        if arch != "pensioner" and stress_months.max() > 0:
            night_per_day = 1.0 + 1.5 * stress_months[m_of_day]
        offs = _nhpp_thinning(rng, day0, n_days, daily, cal.HOUR_CURVE,
                              night_boost_per_day=night_per_day)
        if len(offs) == 0:
            return
        ts = cal.start_us() + day0 * DAY_US + np.sort(offs)
        n = len(ts)
        ev_day = ((ts - cal.start_us()) // DAY_US).astype(np.int64)
        ev_month = cal.month_of_day(ev_day)

        # Per-event MCC: persona mixture with stress-scaled gambling share.
        wts = np.tile(pers_w, (n, 1))
        stress_ev = stress_months[ev_month]
        wts[:, gambling_i] *= 1.0 + 6.0 * stress_ev * (0.3 + tr["risk_appetite"])
        wts = wts / wts.sum(axis=1, keepdims=True)
        cdf = np.cumsum(wts, axis=1)
        mcc_ev = (rng.random((n, 1)) < cdf).argmax(axis=1)

        # Merchant: 78% loyalty pool, else fresh Zipf draw within the MCC.
        merchant_id = np.zeros(n, dtype=np.int64)
        loyal = rng.random(n) < 0.78
        for mi in np.unique(mcc_ev):
            sel = mcc_ev == mi
            ids = np.where(
                loyal[sel],
                pool[mi][rng.integers(0, len(pool[mi]), size=int(sel.sum()))],
                w.merchants.sample_in_mcc(int(mi), rng.random(int(sel.sum()))),
            )
            merchant_id[sel] = ids

        amt = rng.lognormal(w.merchants.mcc_mu[mcc_ev] + mu_shift, w.merchants.mcc_sigma[mcc_ev])
        gambling_ev = mcc_ev == gambling_i
        amt = np.where(gambling_ev, amt * (1.0 + 1.2 * stress_ev), amt)  # chasing losses
        amt = np.round(np.clip(amt * cal.inflation_mult[ev_day], 0.5, 25_000.0), 2)

        online = rng.random(n) < w.merchants.mcc_online_p[mcc_ev] * (0.6 + 0.8 * tr["tech_savviness"])
        contactless = (~online) & (rng.random(n) < 0.65) & (amt < 100)
        channel = np.where(online, "online", np.where(contactless, "contactless", "pos")).astype(object)
        atm = np.array([MCC_KEYS[m] == "atm" for m in mcc_ev])
        channel[atm] = "atm"
        txn_type = np.where(atm, "atm_withdrawal", "card_payment").astype(object)

        m_country = w.merchants.countries[merchant_id]
        # Travel weeks: occasionally spend abroad (more for HNW / high income).
        travel_p = 0.02 + 0.08 * (arch in ("high_net_worth", "trader")) + 0.02 * tr["risk_appetite"]
        n_trips = rng.poisson(travel_p * self.cfg.months)
        abroad = np.zeros(n, dtype=bool)
        for _ in range(int(n_trips)):
            t0 = int(rng.integers(day0, max(day0 + 1, cal.n_days - 8)))
            abroad |= (ev_day >= t0) & (ev_day < t0 + int(rng.integers(3, 9)))
        ev_country = np.where(abroad, m_country, country).astype(object)
        is_fx = np.array([_CCY.get(str(c), "EUR") != ccy for c in ev_country])
        ev_ccy = np.where(is_fx, [_CCY.get(str(c), "EUR") for c in ev_country], ccy).astype(object)

        mcc_codes = np.array([MCC_CATALOG_CODE[MCC_KEYS[m]] for m in mcc_ev], dtype=object)
        names = w.merchants.names[merchant_id]

        txn.append(
            ts,
            amount=_round2str(amt),
            currency=ev_ccy,
            mcc=mcc_codes,
            merchant=names.astype(object),
            txn_type=txn_type,
            channel=channel,
            country=ev_country,
        )
        for t, a in zip(ts.tolist(), amt.tolist()):
            cash.append((t, -a))
        np.add.at(monthly_spend, ev_month, amt)
        np.add.at(monthly_fx, ev_month, np.where(is_fx, amt, 0.0))

        # Organic crypto top-ups for risk-appetite users (after the crypto
        # product launch). Without these, txn_type=crypto_topup exists only in
        # mule cash-outs — a single-field AML marker no realistic book has.
        risk = float(tr["risk_appetite"])
        launch_m = self.world.calendar.product_launch_month.get("crypto", self.cfg.months)
        if risk > 0.5 and launch_m < self.cfg.months:
            d0 = max(day0, int(cal.month_start_day[launch_m]))
            active_days = max(0, cal.n_days - d0)
            n_cx = int(rng.poisson(risk * active_days / 60.0))
            if n_cx:
                cx_ts = cal.start_us() + (d0 * DAY_US +
                                          (rng.random(n_cx) * active_days * DAY_US)).astype(np.int64)
                cx_amt = np.round(np.exp(rng.normal(3.6, 0.9, size=n_cx)), 2)
                txn.append(
                    cx_ts,
                    amount=_round2str(cx_amt),
                    currency=np.full(n_cx, ccy, dtype=object),
                    mcc=np.full(n_cx, MCC_CATALOG_CODE["online_retail"], dtype=object),
                    merchant=_CRYPTO_EXCHANGES[rng.integers(0, len(_CRYPTO_EXCHANGES), size=n_cx)],
                    txn_type=np.full(n_cx, "crypto_topup", dtype=object),
                    channel=np.full(n_cx, "online", dtype=object),
                    country=np.full(n_cx, country, dtype=object),
                )
                for t, a in zip(cx_ts.tolist(), cx_amt.tolist()):
                    cash.append((int(t), -float(a)))
                np.add.at(monthly_spend, cal.month_of_day(np.minimum(
                    (cx_ts - cal.start_us()) // DAY_US, cal.n_days - 1)), cx_amt)

        # Sessions sometimes precede a payment: a fraction of in-app (online)
        # payments are immediately preceded by a short navigation burst that
        # lands on the payments/cards screen.
        online_idx = np.nonzero(online)[0]
        if len(online_idx):
            chosen = online_idx[rng.random(len(online_idx)) < 0.18]
            self._presession(rng, ts[chosen], devices, os_name, country, app)

    def _presession(self, rng, pay_ts: np.ndarray, devices: list[str], os_name: str,
                    country: str, app: SourceBuffer) -> None:
        """Emit a 1–3 screen app burst just before each payment in ``pay_ts``."""
        if len(pay_ts) == 0:
            return
        nav = rng.integers(1, 4, size=len(pay_ts))
        total = int(nav.sum())
        pay_of_ev = np.repeat(pay_ts, nav)
        order_in = np.concatenate([np.arange(k, 0, -1) for k in nav])  # k..1 before payment
        gap = (rng.uniform(15, 90, size=total) * order_in * 1_000_000).astype(np.int64)
        ts = pay_of_ev - gap
        last = np.concatenate([[True] if k == 1 else [False] * (k - 1) + [True] for k in nav])
        screen = np.where(last, "payments", _SCREENS[rng.integers(0, 4, size=total)]).astype(object)
        dev = np.array(devices, dtype=object)[rng.integers(0, len(devices), size=total)]
        app.append(
            ts,
            screen=screen,
            action=np.where(rng.random(total) < 0.6, "view", "tap").astype(object),
            device_id=dev,
            os=np.full(total, os_name, dtype=object),
            app_version=np.full(total, f"10.{rng.integers(0, 40)}", dtype=object),
            geo_country=np.full(total, country, dtype=object),
        )

    # ------------------------------------------------------------------ sessions
    def _sessions(self, rng, spec, tr, act_mult, signup_day, country, app: SourceBuffer,
                  devices: list[str], os_name: str) -> None:
        cal = self.world.calendar
        day0 = signup_day
        n_days = cal.n_days - day0
        if n_days <= 0:
            return
        days = np.arange(day0, cal.n_days)
        m_of_day = cal.month_of_day(days)
        daily = float(spec["session_rate"]) / 30.44 * act_mult[m_of_day] * (0.5 + tr["tech_savviness"])
        evening = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.6, 1.2, 1.6, 1.4, 1.2, 1.3,
                            1.6, 1.3, 1.1, 1.1, 1.3, 1.7, 2.2, 2.4, 2.3, 2.0, 1.4, 0.7])
        evening = evening / evening.mean()
        starts = _nhpp_thinning(rng, day0, n_days, daily, evening)
        if len(starts) == 0:
            return
        starts = cal.start_us() + day0 * DAY_US + np.sort(starts)
        n_sess = len(starts)

        # Hawkes-style bursts: 3–15 navigation events per session over the screen
        # graph. geometric(p) >= 1, so 2 + geometric spans [3, ∞) → clip to 15.
        burst = np.clip(2 + rng.geometric(0.30, size=n_sess), 3, 15)
        total = int(burst.sum())
        sess_of_ev = np.repeat(np.arange(n_sess), burst)
        # Markov walk over screens, vectorized step-by-step across sessions.
        max_len = int(burst.max())
        cur = np.zeros(n_sess, dtype=np.int64)  # all sessions start at home
        screens_steps = np.zeros((max_len, n_sess), dtype=np.int64)
        for step in range(1, max_len):
            u = rng.random(n_sess)
            cur = (u[:, None] < _SCREEN_T_CUM[cur]).argmax(axis=1)
            screens_steps[step] = cur
        pos_in_sess = np.concatenate([np.arange(b) for b in burst])
        screen_idx = screens_steps[pos_in_sess, sess_of_ev]
        gaps = rng.exponential(18.0, size=total) + 2.0  # seconds between screens
        gaps_us = (gaps * 1_000_000).astype(np.int64)
        first = np.zeros(total, dtype=bool)
        first[np.cumsum(burst)[:-1]] = True
        first[0] = True
        gaps_us[first] = 0
        cum = np.cumsum(gaps_us)
        sess_first = np.r_[0, np.cumsum(burst)[:-1]]
        ts = np.repeat(starts, burst) + (cum - np.repeat(cum[sess_first], burst))
        dev = np.array(devices, dtype=object)[rng.integers(0, len(devices), size=total)]
        action = np.where(rng.random(total) < 0.7, "view", "tap").astype(object)
        app.append(
            ts,
            screen=_SCREENS[screen_idx],
            action=action,
            device_id=dev,
            os=np.full(total, os_name, dtype=object),
            app_version=np.full(total, f"10.{rng.integers(0, 40)}", dtype=object),
            geo_country=np.full(total, country, dtype=object),
        )

    # ------------------------------------------------------------------ trading
    def _trading(self, rng, tr, act_mult, signup_day, ccy, trd: SourceBuffer, cash, monthly_trades) -> None:
        cal = self.world.calendar
        if tr["risk_appetite"] < 0.55:
            return
        launch_day = int(cal.month_start_day[cal.product_launch_month["trading"]])
        day0 = max(signup_day, launch_day)
        n_days = cal.n_days - day0
        if n_days <= 0:
            return
        days = np.arange(day0, cal.n_days)
        m_of_day = cal.month_of_day(days)
        rate = 6.0 * (tr["risk_appetite"] - 0.45) * 2.0  # trades / month
        daily = rate / 30.44 * act_mult[m_of_day] * (~cal.is_weekend[days])
        offs = _nhpp_thinning(rng, day0, n_days, daily, cal.HOUR_CURVE)
        if len(offs) == 0:
            return
        ts = cal.start_us() + day0 * DAY_US + np.sort(offs)
        n = len(ts)
        favs = _INSTRUMENTS[rng.choice(len(_INSTRUMENTS), size=min(4, len(_INSTRUMENTS)), replace=False)]
        instr = favs[rng.integers(0, len(favs), size=n)]
        side = np.where(rng.random(n) < 0.58, "buy", "sell").astype(object)
        price = np.round(np.exp(rng.normal(4.2, 1.4, size=n)), 2)
        qty = np.round(np.exp(rng.normal(0.4, 1.0, size=n)), 4)
        notional = price * qty
        trd.append(
            ts,
            instrument=instr.astype(object),
            side=side,
            quantity=np.array([f"{q:.4f}" for q in qty], dtype=object),
            price=_round2str(price),
            order_type=np.where(rng.random(n) < 0.8, "market", "limit").astype(object),
            venue=np.full(n, "internal", dtype=object),
        )
        ev_month = cal.month_of_day(((ts - cal.start_us()) // DAY_US).astype(np.int64))
        np.add.at(monthly_trades, ev_month, notional)
        # Trades settle against a brokerage pot, not the current account, so
        # they don't drive insolvency; fees still feed the LTV profit model.

    # ------------------------------------------------------------------ comms
    def _comms(self, rng, user_idx, tr, lifecycle, signup_day, com: SourceBuffer,
               app: SourceBuffer, devices: list[str], country: str) -> list[tuple[str, int, int, int, int]]:
        cal, camp = self.world.calendar, self.world.campaigns
        rows: list[tuple[str, int, int, int, int]] = []
        engagement = 0.25 + 0.5 * tr["tech_savviness"] + 0.25 * tr["sociability"]
        sent_ts, sent_id, sent_ch, sent_tpl, open_mask, click_mask = [], [], [], [], [], []
        for ci in range(len(camp.campaign_id)):
            t = int(camp.ts_us[ci])
            day = (t - cal.start_us()) // DAY_US
            if day < signup_day + 7:
                continue
            m = int(cal.month_of_day(np.array([day]))[0])
            state = lifecycle[m]
            if state == CHURNED:
                continue
            # Potential outcomes for the 48h-app-open uplift target.
            p0 = float(np.clip(0.10 + 0.55 * STATE_ACTIVITY[state] * engagement, 0.01, 0.95))
            p1 = float(np.clip(p0 + camp.uplift[ci] * engagement * 2.0, 0.01, 0.97))
            u = rng.random()  # shared noise → monotone potential outcomes
            y0, y1 = int(u < p0), int(u < p1)
            treated = int(rng.random() < camp.target_frac[ci])
            rows.append((str(camp.campaign_id[ci]), t, treated, y0, y1))
            if not treated:
                continue
            sent_ts.append(t)
            sent_id.append(str(camp.campaign_id[ci]))
            sent_ch.append(str(camp.channel[ci]))
            sent_tpl.append(str(camp.template[ci]))
            opened = y1 == 1 and rng.random() < 0.8
            open_mask.append(opened)
            click_mask.append(opened and rng.random() < 0.35)
        if sent_ts:
            n = len(sent_ts)
            base = {
                "campaign_id": np.array(sent_id, dtype=object),
                "channel": np.array(sent_ch, dtype=object),
                "template": np.array(sent_tpl, dtype=object),
            }
            com.append(np.array(sent_ts, dtype=np.int64),
                       comm_event=np.full(n, "sent", dtype=object), **base)
            o = np.nonzero(np.asarray(open_mask))[0]
            if len(o):
                dt = (rng.exponential(6.0, size=len(o)) * HOUR_US).astype(np.int64)
                com.append(np.array(sent_ts, dtype=np.int64)[o] + dt,
                           comm_event=np.full(len(o), "open", dtype=object),
                           **{k: v[o] for k, v in base.items()})
                c = np.nonzero(np.asarray(click_mask))[0]
                if len(c):
                    dt2 = dt[np.searchsorted(o, c)] + (rng.uniform(10, 300, size=len(c)) * 1_000_000).astype(np.int64)
                    com.append(np.array(sent_ts, dtype=np.int64)[c] + dt2,
                               comm_event=np.full(len(c), "click", dtype=object),
                               **{k: v[c] for k, v in base.items()})
        return rows

    # ------------------------------------------------------------------ p2p
    def _p2p(self, user_idx, ccy, txn: SourceBuffer, cash) -> None:
        tg = self.world.transfers
        # Ring laundering legs (fan-in/layering/fan-out) live ONLY in the transfer
        # ledger (transfers.parquet → the AML graph), not in the user's card-event
        # stream: laundering moves money via transfers, not distinctive card
        # behaviour. This is what makes AML *relational* — an isolated embedding
        # (built from events) cannot see the ring; only a graph-aware model can.
        out_i = tg.user_outgoing(user_idx)
        in_i = tg.user_incoming(user_idx)
        out_i = out_i[tg.is_mule_leg[out_i] == 0]
        in_i = in_i[tg.is_mule_leg[in_i] == 0]
        if len(out_i):
            ts = tg.ts_us[out_i]
            amt = tg.amount[out_i]
            cps = np.array([f"u_{i:08d}" for i in tg.to_idx[out_i]], dtype=object)
            txn.append(ts, amount=_round2str(amt), currency=np.full(len(ts), ccy, dtype=object),
                       mcc=np.full(len(ts), "4829", dtype=object),
                       merchant=np.full(len(ts), "P2P TRANSFER", dtype=object),
                       txn_type=np.full(len(ts), "p2p_out", dtype=object),
                       channel=np.full(len(ts), "app", dtype=object),
                       country=np.full(len(ts), str(self.world.personas.country[user_idx]), dtype=object),
                       counterparty=cps)
            for t, a in zip(ts.tolist(), amt.tolist()):
                cash.append((t, -a))
        if len(in_i):
            ts = tg.ts_us[in_i]
            amt = tg.amount[in_i]
            cps = np.array([f"u_{i:08d}" for i in tg.from_idx[in_i]], dtype=object)
            txn.append(ts, amount=_round2str(amt), currency=np.full(len(ts), ccy, dtype=object),
                       mcc=np.full(len(ts), "4829", dtype=object),
                       merchant=np.full(len(ts), "P2P TRANSFER", dtype=object),
                       txn_type=np.full(len(ts), "p2p_in", dtype=object),
                       channel=np.full(len(ts), "app", dtype=object),
                       country=np.full(len(ts), str(self.world.personas.country[user_idx]), dtype=object),
                       counterparty=cps)
            for t, a in zip(ts.tolist(), amt.tolist()):
                cash.append((t, a))

    # ------------------------------------------------------------------ premium
    def _maybe_premium(self, rng, tr, income) -> bool:
        p = 0.05 + 0.25 * tr["tech_savviness"] * np.clip(income / 4000.0, 0, 1.5)
        return bool(rng.random() < p)

    # ------------------------------------------------------------------ balance & overdrafts
    def _balance(self, rng, income, txn: SourceBuffer, ccy, cash, signup_day, country="GB") -> tuple[int, float, float]:
        cal = self.world.calendar
        if not cash:
            return -1, 0.0, 0.0
        ts = np.array([c[0] for c in cash], dtype=np.int64)
        amt = np.array([c[1] for c in cash], dtype=np.float64)
        order = np.argsort(ts, kind="stable")
        ts, amt = ts[order], amt[order]
        opening = income * rng.uniform(0.15, 1.1)
        bal = opening + np.cumsum(amt)
        overdraft_limit = -max(500.0, income * 1.25)
        below = bal < 0
        # Overdraft fee events: at most one per day while below zero.
        od_days = np.unique((ts[below] - cal.start_us()) // DAY_US)
        if len(od_days):
            od_days = od_days[:60]  # cap pathological cases
            od_ts = cal.start_us() + od_days * DAY_US + 23 * HOUR_US + 59 * MIN_US
            n = len(od_ts)
            txn.append(od_ts.astype(np.int64),
                       amount=np.full(n, "5.00", dtype=object),
                       currency=np.full(n, ccy, dtype=object),
                       mcc=np.full(n, "6012", dtype=object),
                       merchant=np.full(n, "OVERDRAFT FEE", dtype=object),
                       txn_type=np.full(n, "overdraft_fee", dtype=object),
                       channel=np.full(n, "system", dtype=object),
                       country=np.full(n, country, dtype=object))
        # Insolvency: balance stays below the overdraft limit for 14+
        # consecutive days (deep arrears default even if income later resumes).
        insolvency_day = -1
        deep = bal < overdraft_limit
        if deep.any():
            inf = np.int64(2**62)
            recover_ts = np.where(~deep, ts, inf)
            next_recover = np.minimum.accumulate(recover_ts[::-1])[::-1]
            gap = next_recover[deep] - ts[deep]
            stuck = gap >= 14 * DAY_US
            if stuck.any():
                first = int(ts[deep][np.argmax(stuck)])
                insolvency_day = int((first - cal.start_us()) // DAY_US) + 14
        return insolvency_day, float(bal.min()), float(bal[-1])

    # ------------------------------------------------------------------ merge + profile
    def _merge(
        self, txn, app, trd, com
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[tuple[str, ...], dict[str, np.ndarray]]]]:
        """Time-sort all events into the grouped-dense layout (see UserTrace)."""
        groups: list[tuple[tuple[str, ...], dict[str, np.ndarray]]] = []
        ts_parts: list[np.ndarray] = []
        src_parts: list[np.ndarray] = []
        gid_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        for si, buf in ((0, txn), (1, app), (2, trd), (3, com)):
            for sig, ts, cols in buf.groups():
                gid = len(groups)
                groups.append((sig, cols))
                n = len(ts)
                ts_parts.append(ts)
                src_parts.append(np.full(n, si, dtype=np.int8))
                gid_parts.append(np.full(n, gid, dtype=np.int16))
                row_parts.append(np.arange(n, dtype=np.int32))
        if not ts_parts:
            z = np.zeros(0, dtype=np.int64)
            return z, z.astype(np.int8), z.astype(np.int16), z.astype(np.int32), groups
        ts_all = np.concatenate(ts_parts)
        order = np.argsort(ts_all, kind="stable")
        return (
            ts_all[order],
            np.concatenate(src_parts)[order],
            np.concatenate(gid_parts)[order],
            np.concatenate(row_parts)[order],
            groups,
        )

    def _profile(self, rng, user_idx, arch, tr, income, country, signup_day,
                 is_premium, devices) -> tuple[dict[str, str], list[tuple[str, int]]]:
        cal = self.world.calendar
        age = int(self.world.personas.age[user_idx])
        age_band = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
        income_band = ("0-1k" if income < 1000 else "1k-2k" if income < 2000 else
                       "2k-4k" if income < 4000 else "4k-8k" if income < 8000 else "8k+")
        occupations = {
            # No archetype-exclusive value: occupations are shared across
            # archetypes (the mule maps to the same "employee" as salaried/family)
            # so no label is readable off a single static profile attribute.
            "student": "student", "salaried": "employee", "freelancer": "self_employed",
            "family": "employee", "pensioner": "retired", "high_net_worth": "executive",
            "trader": "self_employed", "dormant": "employee", "mule": "employee",
            "fraud_victim": "employee",
        }
        attributes = {
            "age_band": age_band,
            "country": country,
            "occupation": occupations[arch],
            "income_band": income_band,
            "kyc_tier": "full" if rng.random() < 0.9 else "basic",
            # Premium status is emitted only as the time-stamped `premium_upgrade`
            # lifelong milestone below, never as a static attribute. The milestone
            # is truncatable, so a point-in-time-correct feature sees premium only
            # if the upgrade preceded the cutoff; a static `premium=yes/no` would
            # reflect whole-simulation status and survive truncation, leaking
            # post-eval upgrades into ltv_positive embeddings.
            "marital": "married" if (age > 28 and rng.random() < 0.5) else "single",
            "housing": "rent" if rng.random() < 0.55 else "own",
            "device_count": str(len(devices)),
        }
        s_us = cal.start_us() + signup_day * DAY_US + 10 * HOUR_US
        lifelong: list[tuple[str, int]] = [
            ("account_opened", s_us),
            ("kyc_passed", s_us + int(rng.uniform(0.2, 48) * HOUR_US)),
            ("card_activated", s_us + int(rng.uniform(2, 10) * DAY_US)),
        ]
        if is_premium:
            prem_day = int(rng.integers(signup_day + 30, max(signup_day + 31, cal.n_days)))
            lifelong.append(("premium_upgrade", cal.start_us() + prem_day * DAY_US + 12 * HOUR_US))
        if tr["risk_appetite"] >= 0.55:
            t_day = max(signup_day + 10, int(cal.month_start_day[cal.product_launch_month["trading"]]))
            lifelong.append(("trading_enabled", cal.start_us() + t_day * DAY_US + 9 * HOUR_US))
        if rng.random() < 0.15:
            mv_day = int(rng.integers(signup_day + 60, max(signup_day + 61, cal.n_days)))
            lifelong.append(("address_change", cal.start_us() + mv_day * DAY_US + 14 * HOUR_US))
        lifelong.sort(key=lambda kv: kv[1])
        return attributes, lifelong
