"""Run directories: reproducible bookkeeping for a training run.

A run lives under ``runs/{name}/`` and contains:

- ``run.yaml``      the fully-resolved config;
- ``meta.json``     git hash, library versions, tokenizer hash, seed, hardware;
- ``metrics.jsonl`` one JSON object per logged step (+ optional TensorBoard/wandb);
- ``checkpoints/``  ``last.pt`` + periodic snapshots;
- ``tokenizer/``    a copy of the tokenizer used.

``Run.create`` writes the metadata up front; ``Run.open`` reattaches to an
existing run for ``--resume``.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    import numpy
    import pyarrow
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "cuda": torch.version.cuda or "cpu",
    }


def _hardware() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"platform": platform.platform(), "cpu_count": __import__("os").cpu_count()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    else:
        info["gpu"] = None
    return info


@dataclass
class Run:
    """A handle to a run directory with helpers for config/meta/checkpoints."""

    name: str
    dir: Path

    @property
    def checkpoints(self) -> Path:
        return self.dir / "checkpoints"

    @property
    def tokenizer_dir(self) -> Path:
        return self.dir / "tokenizer"

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.jsonl"

    @classmethod
    def create(
        cls,
        name: str,
        config: dict[str, Any],
        seed: int,
        tokenizer_hash: str,
        runs_root: str | Path = "runs",
        tokenizer_src: str | Path | None = None,
    ) -> Run:
        """Create ``runs/{name}/`` with resolved config + meta; copy the tokenizer."""
        d = Path(runs_root) / name
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        run = cls(name=name, dir=d)
        run.write_config(config)
        meta = {
            "name": name, "git_hash": _git_hash(), "versions": _versions(),
            "tokenizer_hash": tokenizer_hash, "seed": seed, "hardware": _hardware(),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        if tokenizer_src is not None and Path(tokenizer_src).exists():
            import shutil

            run.tokenizer_dir.mkdir(exist_ok=True)
            for f in Path(tokenizer_src).iterdir():
                if f.is_file():
                    shutil.copy2(f, run.tokenizer_dir / f.name)
        return run

    @classmethod
    def open(cls, name: str, runs_root: str | Path = "runs") -> Run:
        """Reattach to an existing run directory (for ``--resume``)."""
        d = Path(runs_root) / name
        if not d.exists():
            raise FileNotFoundError(f"run {name!r} not found under {runs_root}")
        return cls(name=name, dir=d)

    def write_config(self, config: dict[str, Any]) -> None:
        """Persist the resolved config to ``run.yaml``."""
        from omegaconf import OmegaConf

        OmegaConf.save(OmegaConf.create(config), self.dir / "run.yaml")

    def read_config(self) -> dict[str, Any]:
        """Load the resolved config back from ``run.yaml``."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(self.dir / "run.yaml")
        out = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(out, dict)
        return {str(k): v for k, v in out.items()}

    def read_meta(self) -> dict[str, Any]:
        """Load ``meta.json``."""
        return json.loads((self.dir / "meta.json").read_text())

    def last_checkpoint(self) -> Path | None:
        """Path to ``checkpoints/last.pt`` if present."""
        p = self.checkpoints / "last.pt"
        return p if p.exists() else None


def list_runs(runs_root: str | Path = "runs") -> list[dict[str, Any]]:
    """Summarize all runs under ``runs_root`` (for ``pragmatiq runs list``)."""
    root = Path(runs_root)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        last = None
        if (d / "metrics.jsonl").exists():
            lines = (d / "metrics.jsonl").read_text().strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
        out.append({
            "name": d.name, "git_hash": meta.get("git_hash", "")[:8], "seed": meta.get("seed"),
            "gpu": meta.get("hardware", {}).get("gpu"),
            "last_step": (last or {}).get("step"), "last_loss": (last or {}).get("loss"),
        })
    return out
