#!/usr/bin/env python3
"""Project gate-1 credit baseline: hand-crafted features + sklearn HGBT.

Reads a generated dataset directory, builds per-user features STRICTLY from
events before each user's eval point (no leakage), trains a
HistGradientBoostingClassifier on default_12m, and prints ROC-AUC.

Gate 1 expects AUC in ~[0.75, 0.85] at full scale: lower means the generator
carries no causal signal, higher means leakage / separability is unrealistic.

Usage:
    python tests/baselines/credit_gbdt.py --data /path/to/synth [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DAY_US = 86_400_000_000


def build_features(data_dir: Path) -> pd.DataFrame:
    """Per-user features from pre-eval transaction history (causal, no leakage)."""
    labels = pq.read_table(data_dir / "labels" / "default_12m.parquet").to_pandas()
    if labels.empty:
        raise SystemExit("no default_12m labels in dataset")
    eval_us = int(labels["eval_ts"].astype("int64").iloc[0])

    ev = pq.read_table(
        data_dir / "events.parquet", columns=["user_id", "ts", "source", "fields"]
    ).to_pandas()
    ev["ts_us"] = ev["ts"].astype("int64")
    ev = ev[ev["ts_us"] < eval_us]  # strictly pre-eval
    ev = ev[ev["user_id"].isin(set(labels["user_id"]))]

    f = pd.DataFrame([dict(x) for x in ev["fields"]])
    ev = pd.concat([ev.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    txn = ev[ev["source"] == "transaction"].copy()
    txn["amount_f"] = pd.to_numeric(txn["amount"], errors="coerce").fillna(0.0)
    txn["is_salary"] = (txn["txn_type"] == "credit_transfer").astype(int)
    txn["is_gambling"] = (txn["mcc"] == "7995").astype(int)
    txn["is_overdraft"] = (txn["txn_type"] == "overdraft_fee").astype(int)
    txn["is_debit"] = (~txn["txn_type"].isin(["credit_transfer", "p2p_in", "refund"])).astype(int)
    txn["hour"] = (txn["ts_us"] % DAY_US) // 3_600_000_000
    txn["is_night"] = ((txn["hour"] >= 22) | (txn["hour"] < 5)).astype(int)
    txn["m6"] = (txn["ts_us"] >= eval_us - 182 * DAY_US).astype(int)
    txn["m3"] = (txn["ts_us"] >= eval_us - 91 * DAY_US).astype(int)
    txn["m1"] = (txn["ts_us"] >= eval_us - 30 * DAY_US).astype(int)
    txn["debit_amt"] = txn["amount_f"] * txn["is_debit"]
    txn["credit_amt"] = txn["amount_f"] * (1 - txn["is_debit"])
    txn["gamb_amt"] = txn["amount_f"] * txn["is_gambling"]
    txn["sal_amt"] = txn["amount_f"] * txn["is_salary"]
    for w in ("m1", "m3", "m6"):
        txn[f"sal_{w}"] = txn["sal_amt"] * txn[w]
        txn[f"debit_{w}"] = txn["debit_amt"] * txn[w]
        txn[f"gamb_{w}"] = txn["gamb_amt"] * txn[w]
        txn[f"night_{w}"] = txn["is_night"] * txn[w]
        txn[f"od_{w}"] = txn["is_overdraft"] * txn[w]

    g = txn.groupby("user_id")
    cols = {
        "n_txn": g.size(),
        "total_debit": g["debit_amt"].sum(),
        "total_credit": g["credit_amt"].sum(),
        "mean_debit": g["debit_amt"].mean(),
        "max_debit": g["debit_amt"].max(),
        "n_salary": g["is_salary"].sum(),
        "salary_sum": g["sal_amt"].sum(),
        "gambling_total": g["gamb_amt"].sum(),
        "overdraft_total": g["is_overdraft"].sum(),
        "n_merchants": g["merchant"].nunique(),
        "last_txn_age_d": (eval_us - g["ts_us"].max()) / DAY_US,
    }
    for w in ("m1", "m3", "m6"):
        for base in ("sal", "debit", "gamb", "night", "od"):
            cols[f"{base}_{w}"] = g[f"{base}_{w}"].sum()
    feats = pd.DataFrame(cols)
    sal_ts = txn[txn["is_salary"] == 1].groupby("user_id")["ts_us"].max()
    feats["last_salary_age_d"] = ((eval_us - sal_ts) / DAY_US).reindex(feats.index).fillna(400.0)
    feats["net_flow"] = feats["total_credit"] - feats["total_debit"]
    feats["net_m3"] = feats["sal_m3"] - feats["debit_m3"]
    feats["net_m1"] = feats["sal_m1"] - feats["debit_m1"]
    feats["gambling_share_m3"] = feats["gamb_m3"] / feats["debit_m3"].clip(lower=1)
    feats["burn_ratio_m3"] = feats["debit_m3"] / feats["sal_m3"].clip(lower=1)
    # Trend features: recent month vs 6-month average (distress shows as decay).
    feats["sal_trend"] = feats["sal_m1"] / (feats["sal_m6"] / 6).clip(lower=1)
    feats["gamb_trend"] = feats["gamb_m1"] / (feats["gamb_m6"] / 6).clip(lower=1)
    feats["night_trend"] = feats["night_m1"] / (feats["night_m6"] / 6).clip(lower=1)
    feats = feats.fillna(0.0)

    out = labels.merge(feats, left_on="user_id", right_index=True, how="inner")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = build_features(args.data)
    y = df["label"].to_numpy()
    X = df.drop(columns=["user_id", "eval_ts", "label"]).to_numpy(dtype=np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=args.seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=args.seed)
    clf.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    result = {
        "auc": round(auc, 4),
        "n_users_scored": int(len(df)),
        "prevalence": round(float(y.mean()), 5),
        "n_features": int(X.shape[1]),
    }
    print(json.dumps(result, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
