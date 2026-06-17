"""Metric logging: JSONL always, TensorBoard and Weights & Biases optionally.

``MetricLogger`` appends every logged step to ``metrics.jsonl`` (the durable,
dependency-free record) and mirrors scalars to TensorBoard (``runs/{name}/tb``)
when ``torch.utils.tensorboard`` is available and to wandb when enabled and
installed. Missing optional backends degrade to a no-op, never an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricLogger:
    """Append-only JSONL logger with optional TensorBoard / wandb mirrors."""

    def __init__(
        self,
        run_dir: str | Path,
        tensorboard: bool = True,
        wandb: bool = False,
        wandb_project: str = "pragmatiq",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.dir / "metrics.jsonl", "a")
        self._tb = None
        self._wandb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.dir / "tb"))
            except Exception:
                self._tb = None
        if wandb:
            try:
                import wandb as _wandb

                self._wandb = _wandb
                _wandb.init(project=wandb_project, name=run_name, config=config or {},
                            dir=str(self.dir), resume="allow")
            except Exception:
                self._wandb = None

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Record one step's scalars to every active backend."""
        row = {"step": step, **metrics}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def truncate_after(self, step: int) -> None:
        """Drop ``metrics.jsonl`` rows with ``step`` greater than ``step``.

        Called on resume so logged-but-uncheckpointed rows from a crashed
        interval do not duplicate (the JSONL stays monotonic in ``step``). The
        TensorBoard event file is append-only and is not rewritten; its scalars
        are idempotent on re-log of the same step.
        """
        path = self.dir / "metrics.jsonl"
        if not path.exists():
            return
        self._fh.flush()
        kept: list[str] = []
        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    if int(json.loads(line).get("step", -1)) <= step:
                        kept.append(line)
                except (ValueError, json.JSONDecodeError):
                    continue  # drop unparseable / partially-flushed trailing row
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as out:
            out.write("\n".join(kept) + ("\n" if kept else ""))
        tmp.replace(path)
        self._fh.close()
        self._fh = open(path, "a")

    def close(self) -> None:
        """Flush and close all backends."""
        self._fh.close()
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
