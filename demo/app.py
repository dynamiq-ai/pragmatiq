"""pragmatiq demo.

Pick a synthetic user → see their event timeline → the model's user embedding
(attach a fine-tuned head via ``pragmatiq finetune`` for calibrated
fraud / credit / churn scores) → their ego transfer graph.

Run:  streamlit run demo/app.py
Env:  PRAGMATIQ_OUT   (quickstart output dir, default runs/quickstart) — set this to
                      switch the whole layout: the demo reads the trained run from
                      <out>/runs/quickstart and raw data from <out>/raw.
      PRAGMATIQ_RUN / PRAGMATIQ_RAW   (override the run or the raw-data path on their own)

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

# `pragmatiq quickstart --out <OUT>` lays its artifacts out as <OUT>/raw (raw data)
# and <OUT>/runs/quickstart (the trained run), so the demo derives both from
# PRAGMATIQ_OUT and works with no extra env after a default quickstart. Set
# PRAGMATIQ_OUT to switch the whole layout, or override PRAGMATIQ_RUN (the run) and
# PRAGMATIQ_RAW (the data) individually.
OUT = Path(os.environ.get("PRAGMATIQ_OUT", "runs/quickstart"))
# Prefer the quickstart layout (<OUT>/runs/quickstart); fall back to OUT itself so a
# run trained straight into a directory (e.g. `pragmatiq pretrain --name quickstart`
# under the default `runs/` root) is also found without setting PRAGMATIQ_RUN.
_quickstart_run = OUT / "runs" / "quickstart"
RUN = Path(os.environ.get("PRAGMATIQ_RUN") or (_quickstart_run if _quickstart_run.exists() else OUT))
RAW = Path(os.environ.get("PRAGMATIQ_RAW", OUT / "raw"))


@st.cache_resource
def load_model():
    from pragmatiq.models.pragmatiq import PragmaModel

    return PragmaModel.from_pretrained(RUN)


@st.cache_data
def load_events() -> pd.DataFrame:
    df = pq.read_table(RAW / "events.parquet").to_pandas()
    df["fields"] = df["fields"].apply(dict)
    return df


def main() -> None:
    st.set_page_config(page_title="pragmatiq demo", layout="wide")
    st.title("pragmatiq — behavioral foundation model demo")
    st.caption("Independent implementation inspired by the PRAGMA paper (arXiv 2604.08649); "
               "not affiliated with or endorsed by Revolut.")

    if not RUN.exists() or not RAW.exists():
        st.warning(f"Point PRAGMATIQ_RUN ({RUN}) and PRAGMATIQ_RAW ({RAW}) at a trained run "
                   "and a generated dataset. Try `pragmatiq quickstart` first.")
        return

    events = load_events()
    users = sorted(events["user_id"].unique())[:500]
    user_id = st.sidebar.selectbox("Synthetic user", users)
    ev = events[events["user_id"] == user_id].sort_values("ts")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Event timeline — {user_id} ({len(ev)} events)")
        show = ev.copy()
        show["amount"] = show["fields"].apply(lambda f: f.get("amount", ""))
        show["merchant"] = show["fields"].apply(lambda f: f.get("merchant", ""))
        st.dataframe(show[["ts", "source", "amount", "merchant"]].tail(50), use_container_width=True)

    with col2:
        st.subheader("Model scores")
        try:
            model = load_model()
            recs = [{
                "user_id": user_id,
                "events": [(int(pd.Timestamp(r.ts).value // 1000), r.source, dict(r.fields))
                           for r in ev.itertuples()],
                "attributes": {}, "lifelong": [],
            }]
            emb = model.embed_records(recs)
            st.metric("embedding norm", f"{np.linalg.norm(emb[0]):.2f}")
            st.caption("Attach fine-tuned heads (fraud/credit/churn) for calibrated scores; "
                       "this demo shows the raw embedding.")
        except Exception as exc:  # pragma: no cover - demo resilience
            st.error(f"model unavailable: {exc}")

    transfers_path = RAW / "transfers.parquet"
    if transfers_path.exists():
        st.subheader("Ego transfer graph")
        tr = pq.read_table(transfers_path).to_pandas()
        ego = tr[(tr["from_user"] == user_id) | (tr["to_user"] == user_id)]
        st.write(f"{len(ego)} transfers involving {user_id}")
        st.dataframe(ego[["from_user", "to_user", "amount", "ts"]].head(30), use_container_width=True)


if __name__ == "__main__":
    main()
