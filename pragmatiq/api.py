"""Public API: the functions the CLI (and notebooks) call.

Per the global rules, ALL logic lives here or deeper in the library —
``pragmatiq/cli.py`` only parses arguments and calls these functions.

Functions appear here phase by phase:
``synthesize`` (1) · ``tokenize`` (2) · ``pretrain`` (5) · ``finetune`` (5) ·
``embed`` (7) · ``probe`` (5) · ``uplift`` (5).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pragmatiq.core.config import load_yaml as _load_yaml
from pragmatiq.core.env import resolve_device as _resolve_device
from pragmatiq.progress import progress
from pragmatiq.storage.staging import staging as _staging


def _read_shard_tokenizer_hash(shard_dir: str | Path) -> str:
    """Load the live tokenizer hash from a tokenized shard directory."""
    from pragmatiq.data.tokenizer import PragmaTokenizer

    tok_dir = Path(shard_dir) / "tokenizer"
    if not tok_dir.exists():
        raise ValueError(f"{shard_dir!r} is missing tokenizer/; run pragmatiq tokenize first")
    return PragmaTokenizer.load(tok_dir).content_hash


def _run_tokenizer_hash(run: str | Path) -> str:
    """Load the tokenizer hash copied into a training run."""
    from pragmatiq.data.tokenizer import PragmaTokenizer

    return PragmaTokenizer.load(Path(run) / "tokenizer").content_hash


def _ensure_shard_tokenizer_matches_run(shard_dir: str | Path, run: str | Path) -> None:
    """Refuse to combine shards encoded by a different tokenizer than ``run``."""
    shard_hash = _read_shard_tokenizer_hash(shard_dir)
    run_hash = _run_tokenizer_hash(run)
    if shard_hash != run_hash:
        raise ValueError(
            f"tokenizer hash mismatch: shard_dir {shard_hash!r} != run {run_hash!r}. "
            "Re-tokenize with the run tokenizer or use the matching training run."
        )


_RESUME_OPERATIONAL_KEYS = {
    "max_steps",
    "log_every",
    "checkpoint_every_min",
    "verbose",
    "wandb",
    "wandb_project",
}


def _enforce_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Validate that a resumed run keeps architecture/objective config fixed."""
    mismatches = []
    for key in sorted((set(saved) | set(current)) - _RESUME_OPERATIONAL_KEYS):
        if saved.get(key) != current.get(key):
            mismatches.append(f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}")
    if mismatches:
        shown = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            shown += f"; ... +{len(mismatches) - 8} more"
        raise ValueError(
            "resolved config mismatch while resuming; start a new run for architecture, "
            f"optimizer, masking, data, or schedule changes. Differences: {shown}"
        )


def synthesize(
    config: str | Path | dict[str, Any] | None = None,
    out: str | Path = "data/synth",
    n_users: int | None = None,
    seed: int | None = None,
    n_workers: int = 0,
    write_report: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """Generate a synthetic dataset.

    Args:
        config: YAML path or dict of :class:`WorldConfig` fields; ``None`` uses
            defaults.
        out: output directory (events/profiles/transfers/labels parquet).
        n_users / seed: convenience overrides of the corresponding config keys.
        n_workers: parallel sim workers (``<=1`` = inline; output is identical
            regardless).
        write_report: emit ``realism_report.html``.
        **overrides: any further WorldConfig field overrides.

    Returns:
        The generation manifest (also written to ``out/manifest.json``).
    """
    with _staging() as stage:
        out = stage.output(out, is_dir=True)  # type: ignore[assignment]
        from pragmatiq.data.synthetic import WorldConfig, generate

        base: dict[str, Any] = {}
        if isinstance(config, (str, Path)):
            base = _load_yaml(config)
        elif isinstance(config, dict):
            base = dict(config)
        if n_users is not None:
            base["n_users"] = n_users
        if seed is not None:
            base["seed"] = seed
        base.update(overrides)
        cfg = WorldConfig.from_dict(base)
        return generate(cfg, out, n_workers=n_workers, write_report=write_report)


def tokenize(
    data_dir: str | Path,
    out: str | Path,
    config: str | Path | dict[str, Any] | None = None,
    tokenizer_dir: str | Path | None = None,
    max_users: int | None = None,
    rows_per_shard: int = 4096,
    n_workers: int = 0,
) -> dict[str, Any]:
    """Fit (or load) the tokenizer and write tokenized parquet shards + index.

    Args:
        data_dir: a generated dataset (events/profiles/...).
        out: output directory for ``shards/`` + ``user_index.lmdb`` + tokenizer.
        config: tokenizer YAML/dict (defaults if ``None``).
        tokenizer_dir: load an existing tokenizer instead of fitting a new one
            (must match the data's keys; unseen → [UNK]).
        max_users: cap users (debug/quickstart).
        rows_per_shard: parquet rows per shard file.
        n_workers: parallel workers for both the tokenizer fit and the encode
            (<=1 = inline). Workers fold/encode whole row-group ranges; the
            parent merges fit accumulators and stitches boundary users in task
            order, so the fitted tokenizer and the shard files are byte-identical
            for any worker count and any row-group layout (rule 2). Falls back to
            a single process (logged) when the file has <2 usable tasks or an
            oversized row group.

    Returns:
        The shard manifest (also written to ``out/shard_manifest.json``).
    """
    with _staging() as stage:
        data_dir = stage.input(data_dir)  # type: ignore[assignment]
        out = stage.output(out, is_dir=True)  # type: ignore[assignment]
        from pragmatiq.data.sharding import ShardWriter
        from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig, iter_user_records

        data_dir = Path(data_dir)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

        if tokenizer_dir is not None:
            tok = PragmaTokenizer.load(tokenizer_dir)
            tok_src = Path(tokenizer_dir)
        else:
            cfg_dict: dict[str, Any] = {}
            if isinstance(config, (str, Path)):
                cfg_dict = _load_yaml(config)
            elif isinstance(config, dict):
                cfg_dict = dict(config)
            tok = PragmaTokenizer(TokenizerConfig.from_dict(cfg_dict)).fit(data_dir, n_workers=n_workers)
            tok.save(out / "tokenizer")
            tok_src = out / "tokenizer"

        writer = ShardWriter(out, tokenizer_hash=tok.content_hash, rows_per_shard=rows_per_shard)
        # Progress total is best-effort: manifest.json may be absent or foreign
        # (bring-your-own datasets only owe us the parquet contract).
        total: int | None = None
        manifest_path = Path(data_dir) / "manifest.json"
        if manifest_path.exists():
            import json

            try:
                raw = json.loads(manifest_path.read_text()).get("n_users")
                total = int(raw) if raw is not None else None
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                total = None
        if max_users is not None:
            # total == 0 is a real answer (empty dataset), not "unknown"
            total = max_users if total is None else min(total, max_users)
        encoded: Iterator[tuple[Any, dict[str, Any]]]
        if n_workers and n_workers > 1:
            from pragmatiq.data.parallel_tokenize import parallel_tokenize

            encoded = parallel_tokenize(data_dir, tok, tok_src, n_workers, max_users=max_users)
        else:
            records = iter_user_records(data_dir, max_users=max_users)
            encoded = (
                (tok.encode(rec),
                 {"attributes": rec.attributes, "lifelong": rec.lifelong, "as_of": rec.as_of})
                for rec in records
            )
        n = 0
        for enc, profile in progress(encoded, total=total, desc="tokenize", unit="user"):
            writer.add(enc, profile=profile)
            n += 1
        manifest = writer.close()
        manifest["vocab_size"] = tok.vocab_size
        return manifest


def pretrain(
    shard_dir: str | Path,
    run_name: str,
    model_size: str = "small",
    config: str | Path | dict[str, Any] | None = None,
    runs_root: str | Path = "runs",
    resume: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Pretrain a pragmatiq model on tokenized shards.

    Returns a summary dict (run name, final step, last metrics).
    """
    with _staging() as stage:
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        _orig_runs_root = str(runs_root)
        runs_root = stage.output(runs_root, is_dir=True)  # type: ignore[assignment]
        # Resume pre-population: if runs_root was staged (remote) AND the caller
        # explicitly requested resume, materialize the existing remote run dir so
        # the trainer can resume from it.  A fresh run (resume is None or missing)
        # must NOT pull a pre-existing remote run — that would silently inject stale
        # checkpoints into what should be a clean start (see docs/STORAGE.md).
        _runs_root_staged = str(runs_root) != _orig_runs_root
        if _runs_root_staged and resume == "auto":
            from pragmatiq.storage.cache import materialize_dir as _mat_dir
            from pragmatiq.storage.fs import exists as _st_exists

            _remote_run = _orig_runs_root.rstrip("/") + "/" + run_name
            if _st_exists(_remote_run):
                import logging as _logging

                _logging.getLogger(__name__).info(
                    "staging: pre-populating local run dir from %s", _remote_run
                )
                _mat_dir(_remote_run, Path(runs_root) / run_name)
        return _pretrain_inner(shard_dir, run_name, model_size, config, runs_root, resume, **overrides)


def _pretrain_inner(
    shard_dir: str | Path,
    run_name: str,
    model_size: str = "small",
    config: str | Path | dict[str, Any] | None = None,
    runs_root: str | Path = "runs",
    resume: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
    from pragmatiq.data.tokenizer import PragmaTokenizer
    from pragmatiq.experiments.run import Run
    from pragmatiq.experiments.tracking import MetricLogger
    from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel
    from pragmatiq.training.pretrainer import PreTrainer, TrainConfig

    shard_dir = Path(shard_dir)
    tok = PragmaTokenizer.load(shard_dir / "tokenizer")
    base: dict[str, Any] = {}
    # "auto" may arrive as the literal string (Python) or a Path("auto") (the CLI's
    # --config), so normalize before the file-vs-auto decision.
    is_auto = isinstance(config, (str, Path)) and str(config) == "auto"
    if is_auto:
        # Size token_budget / grad_accum / schedule from the data + device so a user can
        # point pretrain at 1M–26M records without tuning. Overrides
        # (incl. num_nodes/devices) still win and are folded in below.
        import logging

        from pragmatiq.training.autoconfig import autoconfigure
        from pragmatiq.training.pretrainer import resolve_device_count

        dev = _resolve_device("auto")
        nodes = int(overrides.get("num_nodes", 1))
        # Respect an explicit `devices` override when sizing the world, so grad_accum and
        # the schedule match the replica count the run will actually use; otherwise size
        # from the visible GPUs. resolve_device_count is the same mapping the trainer uses
        # to build Fabric, so the planned world matches the world the run launches.
        per_node = resolve_device_count(overrides.get("devices", "auto"), dev.startswith("cuda"))
        plan = autoconfigure(shard_dir, device=dev, world_size=nodes * max(1, per_node),
                             model_size=model_size)
        logging.getLogger(__name__).info("auto-config: %s", plan.rationale["note"])
        base = plan.as_overrides()
    elif isinstance(config, (str, Path)):
        base = _load_yaml(config)
    elif isinstance(config, dict):
        base = dict(config)
    base.update(overrides)
    known = set(TrainConfig.__dataclass_fields__) | set(ModelConfig.__dataclass_fields__)
    # Resuming rebuilds the model the checkpoint was trained with (read back from the
    # run's run.yaml), so a run trained at one size resumes at that size rather than
    # the caller's default model_size — otherwise the strict checkpoint load fails on
    # a shape mismatch. The checkpoint's training config is authoritative too — batch
    # sizing, step budget, and sampler position must stay consistent — so an
    # auto-config plan is discarded on resume rather than allowed to re-size the run.
    # Only explicit caller intent applies on top: the `overrides` kwargs always, and
    # an explicitly passed `config` (path or dict) too — e.g. `max_steps=...` or
    # `config={"max_steps": ...}` to extend training.
    resuming = resume == "auto" and (Path(runs_root) / run_name).exists()
    if resuming:
        prev = Run.open(run_name, runs_root).read_config()
        model_size = str(prev.get("model_size", model_size))
        arch = set(ModelConfig.__dataclass_fields__) - {"vocab_size"}
        explicit = dict(overrides) if is_auto else base
        merged = {k: v for k, v in prev.items() if k in known}
        merged.update(explicit)  # explicit caller intent wins for non-architecture knobs
        merged.update({k: prev[k] for k in arch if k in prev})  # architecture stays the checkpoint's
        base = merged
    # Reject unknown config keys so a mistyped option surfaces immediately as a
    # clear error. Recognized keys are TrainConfig or ModelConfig fields; `size`
    # (the model-size selector in configs/model/*.yaml) and the derived
    # `vocab_size` are also accepted.
    unknown = set(base) - known - {"vocab_size", "size"}
    if unknown:
        raise ValueError(f"unknown pretrain config key(s): {sorted(unknown)}; known: {sorted(known)}")
    tcfg = TrainConfig(**{k: v for k, v in base.items() if k in TrainConfig.__dataclass_fields__})
    # Architecture fields (rope_base, dropout, dim, ...) in the config tune the
    # model on top of the size preset (these are the paper-silent knobs the design notes expose).
    model_overrides = {k: v for k, v in base.items()
                       if k in ModelConfig.__dataclass_fields__ and k != "vocab_size"}
    # Hands-off PRAGMA+Nemotron wiring: a tokenizer built in embed mode implies the
    # model needs the matching frozen text encoder (and its width) so text tokens are
    # embedded and reconstructed rather than silently collapsing to [UNK]. An explicit
    # config override still wins.
    if getattr(tok.config, "text_value_mode", "bpe") == "embed":
        model_overrides.setdefault("text_encoder", tok.config.text_encoder)
        model_overrides.setdefault("text_encoder_dim", tok.config.text_encoder_dim)

    from pragmatiq.training.pretrainer import seed_everything

    seed_everything(tcfg.seed, tcfg.deterministic)  # deterministic init + dropout stream (resume-safe)
    model = PragmaModel(ModelConfig.preset(model_size, tok.vocab_size, overrides=model_overrides))
    resolved = {"model_size": model_size, "vocab_size": tok.vocab_size,
                **{k: getattr(model.config, k) for k in
                   ("dim", "n_heads", "depth_profile", "depth_event", "depth_history",
                    "dropout", "rope_base", "max_position", "text_encoder", "text_encoder_dim")},
                **dataclasses.asdict(tcfg)}
    run = (Run.open(run_name, runs_root) if resuming
           else Run.create(run_name, resolved, tcfg.seed, tok.content_hash, runs_root,
                           tokenizer_src=shard_dir / "tokenizer"))
    if resuming:
        _enforce_resume_config(run.read_config(), resolved)
    logger = MetricLogger(run.dir, wandb=tcfg.wandb, wandb_project=tcfg.wandb_project,
                          run_name=run_name, config=resolved)
    trainer = PreTrainer(model, run, tcfg, tok.content_hash, logger=logger)
    ds = ShardDataset(shard_dir)
    sampler = DynamicBatchSampler(ds.index, token_budget=tcfg.token_budget, seed=tcfg.seed)
    loader = ShardDataLoader(ds, sampler)
    trainer.fit(loader, resume=resume)
    logger.close()
    ds.close()
    last = {}
    if run.metrics_path.exists():
        lines = run.metrics_path.read_text().strip().splitlines()
        if lines:
            import json as _json

            last = _json.loads(lines[-1])
    return {"run": run_name, "run_dir": str(run.dir), "steps": trainer.step, "last_metrics": last}


def embed(
    shard_dir: str | Path,
    run: str | Path,
    out: str | Path | None = None,
    token_budget: int = 16_384,
    device: str = "auto",
) -> dict[str, Any]:
    """Embed every user in ``shard_dir`` with a trained model.

    Args:
        shard_dir: tokenized shard directory (from :func:`tokenize`).
        run: run directory of a trained model (loaded via ``from_pretrained``).
        out: if given, write a parquet of ``user_id`` (str) and ``embedding``
            (list<float32>) columns to this path.
        token_budget: per-batch token budget for the embedding passes.
        device: ``"auto"`` (CUDA if present else CPU), ``"cpu"``, or ``"cuda"``.

    Returns:
        ``{"n_users": int, "dim": int}`` — the number of users embedded and the
        embedding dimension.

    Raises:
        ValueError: if no users are found in ``shard_dir``.
    """
    with _staging() as stage:
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        run = stage.input(run)  # type: ignore[assignment]
        out = stage.output(out, is_dir=False) if out is not None else None
        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.models.pragmatiq import PragmaModel
        from pragmatiq.training.probe import embed_users

        device = _resolve_device(device)
        _ensure_shard_tokenizer_matches_run(shard_dir, run)
        model = PragmaModel.from_pretrained(run, device=device)
        ds = ShardDataset(shard_dir)
        emb = embed_users(model, ds, token_budget=token_budget, device=device)
        ds.close()
        if not emb:
            raise ValueError(f"no users found in shard_dir {shard_dir!r}; check the path and that "
                             "tokenize() produced shards")
        if out is not None:
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq

            uids = list(emb)
            mat = np.stack([emb[u] for u in uids])
            table = pa.table({"user_id": uids,
                              "embedding": [row.tolist() for row in mat]})
            pq.write_table(table, Path(out))
        return {"n_users": len(emb), "dim": int(next(iter(emb.values())).shape[0])}


def probe(
    shard_dir: str | Path,
    run: str | Path,
    label_path: str | Path,
    device: str = "auto",
    token_budget: int = 16_384,
    seed: int = 0,
    with_baseline: bool = True,
    probe_model: str = "gbdt",
) -> dict[str, Any]:
    """Probe a trained model on a label table; compares to a raw-count baseline.

    ``probe_model`` selects the head fit on the frozen embedding: ``gbdt`` (default,
    gradient boosting), ``logistic``, or ``lightgbm`` (the ``[gbdt]`` extra). The
    baseline uses the same classifier, so the gap reflects the representation. Both
    ROC-AUC and PR-AUC are returned. Histories are truncated at each user's ``eval_ts``
    (when present) before embedding, for both the probe and the baseline — task metrics
    must never be computed on embeddings that contain the outcome window.
    """
    with _staging() as stage:
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        run = stage.input(run)  # type: ignore[assignment]
        label_path = stage.input(label_path)  # type: ignore[assignment]
        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.models.pragmatiq import PragmaModel
        from pragmatiq.training.probe import (
            EmbeddingProbe,
            RawCountBaseline,
            _load_label_table,
            cutoffs_from_labels,
            embed_users,
        )

        device = _resolve_device(device)
        _ensure_shard_tokenizer_matches_run(shard_dir, run)
        model = PragmaModel.from_pretrained(run, device=device)
        ds = ShardDataset(shard_dir)
        uids, _, eval_us = _load_label_table(label_path)
        cutoffs = cutoffs_from_labels(uids, eval_us)
        emb = embed_users(model, ds, token_budget=token_budget, device=device, cutoffs=cutoffs)
        probe_res = EmbeddingProbe(model=probe_model, seed=seed).run(emb, label_path)
        out_dict: dict[str, Any] = {"probe_model": probe_model,
                                    "probe_auc": probe_res.auc, "probe_pr_auc": probe_res.pr_auc,
                                    "probe_accuracy": probe_res.accuracy,
                                    "n_test": probe_res.n_test, "prevalence": probe_res.prevalence}
        if with_baseline:
            base = RawCountBaseline(seed=seed, model=probe_model).run(ds, label_path)
            out_dict["baseline_auc"] = base.auc
            out_dict["baseline_pr_auc"] = base.pr_auc
        ds.close()
        return out_dict


def uplift(
    shard_dir: str | Path,
    run: str | Path,
    label_path: str | Path,
    device: str = "auto",
    token_budget: int = 16_384,
    seed: int = 0,
    learner: str = "t",
) -> dict[str, Any]:
    """Evaluate communication-campaign uplift on a trained model.

    Embeds users (truncated at each user's first campaign so no campaign-window
    activity leaks in), fits an uplift meta-learner (``learner='t'`` two-model or
    ``'s'`` single-model) on the factual outcomes, and scores predicted uplift
    with the Qini coefficient on a user-grouped held-out split. Reports the
    learner's Qini, the oracle Qini (ranking by the true ``y1 - y0``), and the
    average treatment effect.
    """
    with _staging() as stage:
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        run = stage.input(run)  # type: ignore[assignment]
        label_path = stage.input(label_path)  # type: ignore[assignment]
        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.models.pragmatiq import PragmaModel
        from pragmatiq.training.probe import embed_users
        from pragmatiq.training.uplift import UpliftLearner, cutoffs_from_uplift, load_comm_uplift

        device = _resolve_device(device)
        _ensure_shard_tokenizer_matches_run(shard_dir, run)
        model = PragmaModel.from_pretrained(run, device=device)
        ds = ShardDataset(shard_dir)
        rows = load_comm_uplift(label_path)
        cutoffs = cutoffs_from_uplift(rows)
        emb = embed_users(model, ds, token_budget=token_budget, device=device, cutoffs=cutoffs)
        res = UpliftLearner(seed=seed, learner=learner).run(emb, rows)
        ds.close()
        return {"qini": res.qini, "qini_oracle": res.qini_oracle, "ate": res.ate,
                "n_train": res.n_train, "n_test": res.n_test, "treated_frac": res.treated_frac}


def finetune(
    shard_dir: str | Path,
    run: str | Path,
    label_path: str | Path,
    config: str | Path | dict[str, Any] | None = None,
    device: str = "auto",
    **overrides: Any,
) -> dict[str, Any]:
    """LoRA fine-tune a trained model's adapters + head on a label table.

    The backbone is frozen; only the injected LoRA factors and the task head train,
    with early stopping on a held-out split. Histories are truncated at each user's
    label eval point so the fine-tune never sees the outcome window.

    Args:
        shard_dir: tokenized shard directory.
        run: run directory of the pretrained backbone.
        label_path: parquet label table (``user_id``, ``label``, optional ``eval_ts``).
        config: a ``FineTuneConfig`` YAML path or dict; unknown keys raise.
        device: ``"auto"`` (CUDA if present else CPU), ``"cpu"``, or ``"cuda"``.
        **overrides: ``FineTuneConfig`` fields that override ``config``.

    Returns:
        ``{"best_val_auc", "epochs_run", "n_adapted", "val_auc_history"}`` — best
        held-out AUC, epochs run before early-stopping, number of injected LoRA
        adapters, and the per-epoch validation-AUC history.
    """
    with _staging() as stage:
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        run = stage.input(run)  # type: ignore[assignment]
        label_path = stage.input(label_path)  # type: ignore[assignment]
        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.models.pragmatiq import PragmaModel
        from pragmatiq.training.finetuner import FineTuneConfig, LoRAFineTuner
        from pragmatiq.training.pretrainer import seed_everything

        base: dict[str, Any] = {}
        if isinstance(config, (str, Path)):
            base = _load_yaml(config)
        elif isinstance(config, dict):
            base = dict(config)
        base.update(overrides)
        unknown = set(base) - set(FineTuneConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown finetune config key(s): {sorted(unknown)}; "
                             f"known: {sorted(FineTuneConfig.__dataclass_fields__)}")
        fcfg = FineTuneConfig(**{k: v for k, v in base.items()
                                 if k in FineTuneConfig.__dataclass_fields__})
        # Seed before LoRA init + dropout so the same seed → identical adapters (rule 2).
        seed_everything(fcfg.seed)
        device = _resolve_device(device)
        _ensure_shard_tokenizer_matches_run(shard_dir, run)
        model = PragmaModel.from_pretrained(run, device=device)
        ds = ShardDataset(shard_dir)
        result = LoRAFineTuner(model, fcfg, device=device).fit(ds, label_path)
        ds.close()
        return result


def export(
    run: str | Path,
    shard_dir: str | Path,
    out: str | Path = "pragmatiq_embedder.onnx",
    device: str = "cpu",
) -> dict[str, Any]:
    """Export the dense ONNX reformulation of the model from one example user.

    ONNX export runs on CPU (the dense graph and its constants are built on CPU and
    validated against onnxruntime's CPU provider); ``device`` must be ``"cpu"``.
    """
    with _staging() as stage:
        run = stage.input(run)  # type: ignore[assignment]
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        out = stage.output(out, is_dir=False)  # type: ignore[assignment]
        from pragmatiq.data.collate import VarlenCollator
        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.inference.export import export_onnx
        from pragmatiq.models.pragmatiq import PragmaModel

        if _resolve_device(device) != "cpu":
            raise ValueError(f"ONNX export runs on CPU; pass device='cpu' (got {device!r})")
        _ensure_shard_tokenizer_matches_run(shard_dir, run)
        model = PragmaModel.from_pretrained(run, device="cpu")
        ds = ShardDataset(shard_dir)
        example = VarlenCollator()([ds.get(ds.user_ids[0])])
        ds.close()
        return export_onnx(model, example, out)


def benchmark(
    run: str | Path,
    shard_dir: str | Path,
    device: str = "auto",
    out: str | Path = "deploy/benchmarks/RESULTS.md",
    max_users: int | None = None,
) -> dict[str, Any]:
    """Benchmark batch-embedding throughput and write RESULTS.md."""
    with _staging() as stage:
        run = stage.input(run)  # type: ignore[assignment]
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        out = stage.output(out, is_dir=False)  # type: ignore[assignment]
        from pragmatiq.inference.benchmark import benchmark_batch_embed, write_results
        from pragmatiq.models.pragmatiq import PragmaModel

        device = _resolve_device(device)
        _ensure_shard_tokenizer_matches_run(shard_dir, run)
        model = PragmaModel.from_pretrained(run, device=device)
        stats = benchmark_batch_embed(model, shard_dir, device=device, max_users=max_users)
        write_results(stats, out)
        return stats


def gnn(
    shard_dir: str | Path,
    run: str | Path,
    transfers_path: str | Path,
    aml_label_path: str | Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    device: str = "auto",
    epochs: int = 150,
) -> dict[str, Any]:
    """Run the four-arm AML GNN ablation.

    Embeds users with a trained model, builds the transfer graph, and compares
    (a) isolated embeddings, (b) GraphSAGE+PRAGMA, (c) GraphSAGE+handcrafted.

    AML is full-observation mule-ring membership detection, not a forecast: each
    user is embedded from their entire observed history with no eval-point
    truncation. The label table's ``observed_through`` column records the horizon
    that history was observed through (not a point-in-time cut), so the embedding
    timestamp is ignored here. See MODEL_CARD.md.
    """
    with _staging() as stage:
        shard_dir = stage.input(shard_dir)  # type: ignore[assignment]
        run = stage.input(run)  # type: ignore[assignment]
        transfers_path = stage.input(transfers_path)  # type: ignore[assignment]
        aml_label_path = stage.input(aml_label_path)  # type: ignore[assignment]
        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.models.gnn import run_aml_ablation
        from pragmatiq.models.pragmatiq import PragmaModel
        from pragmatiq.training.probe import _load_label_table, embed_users

        device = _resolve_device(device)
        _ensure_shard_tokenizer_matches_run(shard_dir, run)
        model = PragmaModel.from_pretrained(run, device=device)
        ds = ShardDataset(shard_dir)
        emb = embed_users(model, ds, device=device)
        uids, labels, _ = _load_label_table(aml_label_path)
        label_map = {u: int(lab) for u, lab in zip(uids, labels)}
        ds.close()
        return run_aml_ablation(transfers_path, emb, label_map, seeds=seeds, epochs=epochs)


def validate(data_dir: str | Path) -> dict[str, Any]:
    """Validate a raw dataset against the data contract."""
    with _staging() as stage:
        data_dir = stage.input(data_dir)  # type: ignore[assignment]
        from pragmatiq.validate import validate_dataset

        report = validate_dataset(data_dir)
        return {"ok": report.ok, "errors": report.errors, "warnings": report.warnings,
                "summary": report.summary()}


def quickstart(
    out: str | Path = "runs/quickstart",
    n_users: int = 50_000,
    seed: int = 0,
    model_size: str = "nano",
    max_steps: int = 400,
    n_workers: int = 0,
) -> dict[str, Any]:
    """End-to-end smoke: synth → tokenize → nano pretrain → probe.

    CPU-capable and self-contained; ``max_steps``/``n_users`` are sized for a short run.

    Returns:
        ``{"run_dir", "probe", "message"}`` — the pretrain run directory, the full
        probe result dict (``probe_auc``/``baseline_auc``/...), and a one-line
        human summary of the credit probe AUC vs the raw-count baseline.
    """
    out = Path(out)
    raw, tok = out / "raw", out / "tok"
    synthesize({"n_users": n_users, "seed": seed}, out=raw, n_workers=n_workers, write_report=False)
    tokenize(raw, tok, config={"target_vocab": 28000, "n_buckets": 64})
    summary = pretrain(tok, "quickstart", model_size=model_size,
                       config={"max_steps": max_steps, "token_budget": 8192,
                               "warmup_steps": max(10, max_steps // 10)},
                       runs_root=out / "runs")
    res = probe(tok, summary["run_dir"], raw / "labels" / "default_12m.parquet")
    return {"run_dir": summary["run_dir"], "probe": res,
            "message": f"credit probe AUC {res['probe_auc']:.3f} vs raw-count baseline "
                       f"{res['baseline_auc']:.3f}"}


def runs_list(runs_root: str | Path = "runs") -> list[dict[str, Any]]:
    """List runs under ``runs_root`` with their last logged step/loss/metrics."""
    with _staging() as stage:
        runs_root = stage.input(runs_root)  # type: ignore[assignment]
        from pragmatiq.experiments.run import list_runs

        return list_runs(runs_root)


def runs_compare(names: list[str], runs_root: str | Path = "runs") -> list[dict[str, Any]]:
    """Compare several runs' last metrics side by side (missing runs flagged)."""
    with _staging() as stage:
        runs_root = stage.input(runs_root)  # type: ignore[assignment]
        from pragmatiq.experiments.compare import compare_runs

        return compare_runs(names, runs_root)


def calibrate(
    stats: str | Path,
    config: str | Path | dict[str, Any] | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Fit generator priors to bank-shareable aggregate statistics.

    Returns the calibrated WorldConfig as a dict; if ``out`` is given, also
    writes it as YAML ready for ``pragmatiq synth --config``.
    """
    with _staging() as stage:
        stats = stage.input(stats)  # type: ignore[assignment]
        out = stage.output(out, is_dir=False) if out is not None else None
        from pragmatiq.data.synthetic.calibrate import calibrate_config

        base: dict[str, Any] = {}
        if isinstance(config, (str, Path)):
            base = _load_yaml(config)
        elif isinstance(config, dict):
            base = dict(config)
        result = calibrate_config(_load_yaml(stats), base)
        if out is not None:
            from omegaconf import OmegaConf

            OmegaConf.save(OmegaConf.create(result), Path(out))
        return result
