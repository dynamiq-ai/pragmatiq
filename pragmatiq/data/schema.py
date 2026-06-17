"""Raw data contract: Arrow schemas and record types shared across pragmatiq.

The on-disk contract (see the internal spec "Data contract") is four parquet files:

- ``events.parquet``:    user_id, ts (timestamp[us]), source, fields (map<str,str>)
- ``profiles.parquet``:  user_id, as_of, attributes (map<str,str>),
  lifelong (list<struct<key, ts>>)
- ``transfers.parquet``: from_user, to_user, ts, amount   (optional, for the GNN)
- ``labels/*.parquet``:  task tables keyed by user_id (+ eval point)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pyarrow as pa

log = logging.getLogger(__name__)
_warned_naive_ts = False

SOURCES: tuple[str, ...] = ("transaction", "app", "trading", "communication")

EVENTS_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("ts", pa.timestamp("us"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("fields", pa.map_(pa.string(), pa.string()), nullable=False),
    ]
)

PROFILES_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("as_of", pa.timestamp("us"), nullable=False),
        pa.field("attributes", pa.map_(pa.string(), pa.string()), nullable=False),
        pa.field(
            "lifelong",
            pa.list_(
                pa.struct([pa.field("key", pa.string()), pa.field("ts", pa.timestamp("us"))])
            ),
            nullable=False,
        ),
    ]
)

TRANSFERS_SCHEMA = pa.schema(
    [
        pa.field("from_user", pa.string(), nullable=False),
        pa.field("to_user", pa.string(), nullable=False),
        pa.field("ts", pa.timestamp("us"), nullable=False),
        pa.field("amount", pa.float64(), nullable=False),
    ]
)


def label_schema(task: str) -> pa.Schema:
    """Arrow schema for a label table. All tables carry user_id; extras vary by task."""
    base = [pa.field("user_id", pa.string(), nullable=False)]
    if task in ("default_12m", "churn_6m", "ltv_positive", "aml"):
        cols = base + [
            pa.field("eval_ts", pa.timestamp("us"), nullable=False),
            pa.field("label", pa.int8(), nullable=False),
        ]
        if task == "ltv_positive":
            cols.append(pa.field("profit_6m", pa.float64(), nullable=False))
        return pa.schema(cols)
    if task == "fraud":
        return pa.schema(
            base
            + [pa.field("ts", pa.timestamp("us"), nullable=False), pa.field("label", pa.int8(), nullable=False)]
        )
    if task == "recurring":
        return pa.schema(
            base
            + [
                pa.field("ts", pa.timestamp("us"), nullable=False),
                pa.field("series_id", pa.string(), nullable=False),
                pa.field("label", pa.int8(), nullable=False),
            ]
        )
    if task == "comm_uplift":
        return pa.schema(
            base
            + [
                pa.field("campaign_id", pa.string(), nullable=False),
                pa.field("ts", pa.timestamp("us"), nullable=False),
                pa.field("treated", pa.int8(), nullable=False),
                pa.field("y0", pa.int8(), nullable=False),
                pa.field("y1", pa.int8(), nullable=False),
            ]
        )
    raise ValueError(f"unknown label task {task!r}")


LABEL_TASKS: tuple[str, ...] = (
    "default_12m",
    "fraud",
    "churn_6m",
    "ltv_positive",
    "recurring",
    "aml",
    "comm_uplift",
)


@dataclass
class UserRecord:
    """One user's full raw history, the unit consumed by the tokenizer.

    ``events`` is a list of ``(ts_us, source, fields)`` tuples sorted by time;
    ``attributes`` are static profile key→value pairs; ``lifelong`` is a list of
    ``(key, ts_us)`` profile milestones (account opened, KYC passed, ...).
    """

    user_id: str
    events: list[tuple[int, str, dict[str, str]]] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    lifelong: list[tuple[str, int]] = field(default_factory=list)
    as_of: int = 0  # profile snapshot timestamp, µs since epoch

    @classmethod
    def from_dict(cls, d: dict) -> UserRecord:
        """Build a record from a plain dict (the notebook/serving format).

        Accepts ``events`` as dicts ``{"ts", "source", "fields"}`` or as
        ``(ts, source, fields)`` tuples, and ``lifelong`` as dicts
        ``{"key", "ts"}`` or ``(key, ts)`` tuples. ``ts`` may be an int (µs
        since epoch) or a datetime. Field values are coerced to ``str`` so
        unseen/odd types tokenize to ``[UNK]`` rather than raising.
        """
        import datetime as _dt

        def to_us(ts: object) -> int:
            if isinstance(ts, (int, float)):
                return int(ts)
            if isinstance(ts, _dt.datetime):
                if ts.tzinfo:
                    t = ts
                else:
                    global _warned_naive_ts
                    if not _warned_naive_ts:
                        log.warning(
                            "naive datetime encountered; assuming UTC. Pass timezone-aware "
                            "datetimes for local-time data (and set the tokenizer's calendar_tz "
                            "for local calendar features)."
                        )
                        _warned_naive_ts = True
                    t = ts.replace(tzinfo=_dt.UTC)
                # exact integer µs (avoid float-seconds rounding vs timestamp[us])
                return (t - _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)) // _dt.timedelta(microseconds=1)
            raise TypeError(f"event/lifelong ts must be int µs or datetime, got {type(ts).__name__}")

        events: list[tuple[int, str, dict[str, str]]] = []
        for e in d.get("events", []):
            if isinstance(e, dict):
                fields = {str(k): str(v) for k, v in dict(e.get("fields", {})).items()}
                events.append((to_us(e["ts"]), str(e["source"]), fields))
            else:
                ts, source, fields = e
                events.append((to_us(ts), str(source), {str(k): str(v) for k, v in dict(fields).items()}))
        lifelong: list[tuple[str, int]] = []
        for item in d.get("lifelong", []):
            if isinstance(item, dict):
                lifelong.append((str(item["key"]), to_us(item["ts"])))
            else:
                lifelong.append((str(item[0]), to_us(item[1])))
        return cls(
            user_id=str(d["user_id"]),
            events=events,
            attributes={str(k): str(v) for k, v in dict(d.get("attributes", {})).items()},
            lifelong=lifelong,
            as_of=to_us(d["as_of"]) if d.get("as_of") else 0,
        )
