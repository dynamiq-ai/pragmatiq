"""LabelOracle: turn simulated traces + latent logs into task label tables.

Leakage rules:

- user-level labels (default_12m, churn_6m, ltv_positive) are computed ONLY
  from what happens strictly AFTER the task's eval point;
- eligibility (who gets a row) is computed ONLY from what happens BEFORE it;
- event-level labels (fraud, recurring) are exact event memberships;
- aml is mule-ring membership; comm_uplift stores both potential outcomes.

A small ``label_noise`` flips user-level binaries to keep ceilings realistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pragmatiq.core.schema import SOURCE_TXN
from .config import WorldConfig
from .simulator import UserTrace
from .world import DAY_US, World


@dataclass
class LabelRows:
    """Accumulated label rows for a block of users (columnar lists)."""

    default_12m: list[tuple[str, int, int]] = field(default_factory=list)  # (uid, eval_ts, y)
    fraud: list[tuple[str, int, int]] = field(default_factory=list)  # (uid, ts, 1)
    churn_6m: list[tuple[str, int, int]] = field(default_factory=list)
    ltv_positive: list[tuple[str, int, int, float]] = field(default_factory=list)
    recurring: list[tuple[str, int, str, int]] = field(default_factory=list)  # (uid, ts, series, 1)
    aml: list[tuple[str, int, int]] = field(default_factory=list)  # (uid, observed_through, is_mule)
    comm_uplift: list[tuple[str, str, int, int, int, int]] = field(default_factory=list)

    def extend(self, other: LabelRows) -> None:
        """Merge another block's rows into this one (order-preserving)."""
        self.default_12m.extend(other.default_12m)
        self.fraud.extend(other.fraud)
        self.churn_6m.extend(other.churn_6m)
        self.ltv_positive.extend(other.ltv_positive)
        self.recurring.extend(other.recurring)
        self.aml.extend(other.aml)
        self.comm_uplift.extend(other.comm_uplift)


class LabelOracle:
    """Computes all task labels for one user from the trace and latent log."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.cfg: WorldConfig = world.cfg
        cal = world.calendar
        self.eval_credit_day = int(cal.month_start_day[self.cfg.eval_month_credit])
        self.eval_short_day = int(cal.month_start_day[self.cfg.eval_month_short])
        self.eval_credit_us = cal.start_us() + self.eval_credit_day * DAY_US
        self.eval_short_us = cal.start_us() + self.eval_short_day * DAY_US
        self.observed_through_us = cal.start_us() + cal.n_days * DAY_US
        # End of the 12-month outcome window, taken at the month boundary 12
        # months after the credit eval point. __post_init__ guarantees
        # eval_month_credit + 12 <= months, so this stays inside the simulated
        # horizon — the positive prevalence is a true 12-month outcome and is not
        # biased low by a fixed day count overshooting the end of the simulation.
        self.eval_credit_end_day = int(cal.month_start_day[self.cfg.eval_month_credit + 12])

    def label_user(self, trace: UserTrace, rng: np.random.Generator) -> LabelRows:
        """Produce this user's rows for every label table.

        ``rng`` is the user's own generator (continued after simulation) so
        label noise stays per-user deterministic.
        """
        cfg = self.cfg
        rows = LabelRows()
        uid = trace.user_id
        noise = cfg.label_noise

        txn_ts = trace.ts[trace.source == SOURCE_TXN] if len(trace.ts) else np.zeros(0, dtype=np.int64)

        # ---- default_12m: eligibility = active in the 90d before credit eval
        # AND not already insolvent at eval (banks don't score charged-off books);
        # outcome = insolvency within 12m AFTER eval.
        pre = (txn_ts >= self.eval_credit_us - 90 * DAY_US) & (txn_ts < self.eval_credit_us)
        already_insolvent = 0 <= trace.insolvency_day <= self.eval_credit_day
        if pre.sum() >= 3 and not already_insolvent:
            y = 0
            if trace.insolvency_day >= 0:
                d = trace.insolvency_day
                if self.eval_credit_day < d <= self.eval_credit_end_day:
                    y = 1
            if rng.random() < noise:
                y = 1 - y
            rows.default_12m.append((uid, self.eval_credit_us, y))

        # ---- fraud: txn-level positives inside FraudEpisodes, plus up to 3
        # sampled clean transactions per user as negatives — a positives-only
        # table cannot train or score any binary classifier.
        for t in trace.fraud_rows:
            rows.fraud.append((uid, int(t), 1))
        if len(txn_ts):
            fraud_set = np.asarray(trace.fraud_rows, dtype=np.int64)
            clean = txn_ts[~np.isin(txn_ts, fraud_set)] if len(fraud_set) else txn_ts
            k = min(3, len(clean))
            if k:
                for t in rng.choice(clean, size=k, replace=False):
                    rows.fraud.append((uid, int(t), 0))

        # ---- churn_6m: eligibility = signed up and still active (not churned)
        # at the short eval point; outcome = churns within 6m AFTER eval.
        m_eval = cfg.eval_month_short
        signed_up = len(trace.ts) > 0 and int((trace.ts < self.eval_short_us).sum()) >= 1
        active_at_eval = trace.churn_month == -1 or trace.churn_month > m_eval
        if signed_up and active_at_eval:
            horizon = min(cfg.months, m_eval + 6)
            churned_after = trace.churn_month != -1 and m_eval < trace.churn_month <= horizon
            y = int(churned_after)
            if rng.random() < noise:
                y = 1 - y
            rows.churn_6m.append((uid, self.eval_short_us, y))

            # ---- ltv_positive shares the same eligible population: 6m gross profit
            # AFTER the short eval point (interchange + premium + trading + FX − servicing).
            sl = slice(m_eval, horizon)
            interchange = 0.007 * float(trace.monthly_spend[sl].sum())
            premium_fee = 6.99 * (horizon - m_eval) if trace.is_premium else 0.0
            trading_fee = 0.001 * float(trace.monthly_trades[sl].sum())
            fx_markup = 0.01 * float(trace.monthly_fx[sl].sum())
            servicing = 3.0 * (horizon - m_eval)
            profit = interchange + premium_fee + trading_fee + fx_markup - servicing
            y_ltv = int(profit > 0)
            if rng.random() < noise:
                y_ltv = 1 - y_ltv
            rows.ltv_positive.append((uid, self.eval_short_us, y_ltv, round(profit, 2)))

        # ---- recurring: exact series membership, plus sampled non-recurring
        # transactions as negatives (series id "" by convention).
        for ts, sid in trace.recurring_rows:
            rows.recurring.append((uid, int(ts), sid, 1))
        if len(txn_ts):
            rec_ts = np.asarray([t for t, _ in trace.recurring_rows], dtype=np.int64)
            non_rec = txn_ts[~np.isin(txn_ts, rec_ts)] if len(rec_ts) else txn_ts
            k = min(3, len(non_rec))
            if k:
                for t in rng.choice(non_rec, size=k, replace=False):
                    rows.recurring.append((uid, int(t), "", 0))

        # ---- aml: full-observation mule-ring membership, one row per user. The
        # label reflects the user's role over their whole observed history, so the
        # time column records the horizon that history was observed through rather
        # than a forecast cut-off.
        is_mule = bool(self.world.episodes.mule_member[trace.user_idx])
        rows.aml.append((uid, self.observed_through_us, int(is_mule)))

        # ---- comm_uplift: both potential outcomes per (user, campaign).
        for cid, ts, treated, y0, y1 in trace.comm_rows:
            rows.comm_uplift.append((uid, cid, int(ts), treated, y0, y1))

        return rows
