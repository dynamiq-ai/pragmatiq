"""Masked-LM pretrainer on Lightning Fabric.

``PreTrainer.fit`` runs the MLM objective with:

- bf16 mixed precision on GPU, fp32 on CPU (Fabric handles device/DDP);
- Muon (2-D hidden weights) + AdamW (embeddings/norms/biases), grad clip 1.0,
  cosine schedule with warmup;
- a checkpoint every ``checkpoint_every_min`` minutes capturing model, BOTH
  optimizers, scheduler, sampler position, RNG states (torch/numpy/cuda + the
  masking generator), tokenizer hash and resolved config (global rule 3);
- ``resume="auto"`` picks up ``checkpoints/last.pt`` and reproduces the exact
  batch + masking stream;
- NaN/inf loss → dump the batch to ``debug/`` and skip the step;
- per-step logging of total loss, per-masking-type loss, MLM accuracy, grad
  norm, LR, tokens/sec, and GPU memory.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.collate import PackedBatch
from ..data.dataset import ShardDataLoader
from ..experiments.run import Run
from ..experiments.tracking import MetricLogger
from ..models.heads import MLMHead, mlm_loss, text_mse_loss
from ..models.pragmatiq import CKPT_FORMAT, PragmaModel
from ..registry import get_masker
from .masking import TYPE_NAMES, MaskingStrategy
from .optim import WarmupCosine, build_optimizers

log = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python/NumPy/torch RNGs so a run is reproducible from ``seed``.

    Call this before constructing the model so weight init and the dropout
    stream are deterministic (the resume test relies on it).

    ``deterministic`` is a symmetric, process-wide toggle. With ``True`` it opts
    into deterministic CUDA kernels: it sets ``CUBLAS_WORKSPACE_CONFIG``, enables
    ``torch.use_deterministic_algorithms``, pins cuDNN to its deterministic
    (non-benchmarked) path, and exports ``PRAGMATIQ_DETERMINISTIC=1`` so the
    attention layer selects flash-attn's deterministic backward. With ``False``
    (the default) it clears those switches, so a non-deterministic call always
    restores the standard, faster path even if an earlier call enabled
    determinism in the same process. See :attr:`TrainConfig.deterministic` for
    the precision coupling.

    CUDA-only side effects (H100 run findings 2026-06):

    - ``TORCH_NCCL_ASYNC_ERROR_HANDLING`` and ``NCCL_ASYNC_ERROR_HANDLING`` are set
      to ``1`` via ``os.environ.setdefault`` (no override of user-set values) so that
      if a rank crashes mid-collective (e.g. OOM), the NCCL call aborts instead of all
      ranks hanging forever (observed deadlock on 4-GPU `large` run).
    - ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (setdefault) reduces
      fragmentation-driven OOM by allowing the allocator to expand existing segments
      rather than requesting new ones. The H100 OOM error explicitly suggested this.
    - ``torch.set_float32_matmul_precision("high")`` is set when CUDA is available
      and ``deterministic=False`` to enable TF32 Tensor Core matmuls (free speedup on
      Ampere/Hopper). In deterministic mode ``"highest"`` (default) is preserved so
      fp32 reproducibility is not compromised.
    """
    import os
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # NCCL fail-fast: if a rank dies (e.g. OOM), abort the collective instead of
        # hanging all other ranks. Use setdefault so a user-set value is not overridden.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")  # legacy alias
        # Reduce fragmentation-driven OOM by reusing existing segment memory.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["PRAGMATIQ_DETERMINISTIC"] = "1"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Reset fp32 matmul precision to "highest" so the switch is SYMMETRIC: a
        # deterministic call after a non-deterministic one (which set "high"/TF32) in
        # the same process must restore full-precision fp32 matmuls, or the "bit-exact
        # in fp32" determinism guarantee silently leaks TF32 (an approximation).
        torch.set_float32_matmul_precision("highest")
    else:
        # Symmetric off-switch: a non-deterministic call clears any deterministic
        # state a prior call set, so the process never sticks on the slow path.
        # (CUBLAS_WORKSPACE_CONFIG is inert once the algorithm switch is off.)
        os.environ.pop("PRAGMATIQ_DETERMINISTIC", None)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            # TF32 Tensor Core matmuls: free ~10-20% speedup on Ampere/Hopper with
            # negligible numerical impact (bf16 training already accepts that tolerance).
            # Only set in non-deterministic mode; "highest" is the default so
            # deterministic mode continues to produce bit-exact fp32 results.
            torch.set_float32_matmul_precision("high")


def _pack_numpy_state(state: Any) -> dict[str, Any]:
    """Convert ``np.random.get_state()`` to a torch ``weights_only``-safe form."""
    name, keys, pos, has_gauss, cached = state
    return {"name": name, "keys": torch.from_numpy(keys.astype("int64")),
            "pos": int(pos), "has_gauss": int(has_gauss), "cached": float(cached)}


def _unpack_numpy_state(d: dict[str, Any]) -> tuple:
    """Inverse of :func:`_pack_numpy_state`."""
    keys = d["keys"].numpy().astype("uint32")
    return (d["name"], keys, d["pos"], d["has_gauss"], d["cached"])


_CHECKPOINT_OPERATIONAL_KEYS = {
    "max_steps",
    "log_every",
    "checkpoint_every_min",
    "verbose",
    "wandb",
    "wandb_project",
}


def _require_matching_config(
    name: str,
    saved: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    ignored: set[str] | None = None,
) -> None:
    """Validate checkpoint-embedded config against the current object config."""
    if saved is None:
        raise ValueError(f"checkpoint missing {name}")
    ignored = ignored or set()
    mismatches = []
    for key in sorted((set(saved) | set(current)) - ignored):
        if saved.get(key) != current.get(key):
            mismatches.append(f"{key}: checkpoint={saved.get(key)!r} current={current.get(key)!r}")
    if mismatches:
        shown = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            shown += f"; ... +{len(mismatches) - 8} more"
        raise ValueError(f"checkpoint {name} mismatch: {shown}")


@dataclasses.dataclass
class TrainConfig:
    """Pretraining hyperparameters (GUESS defaults per SPEC, all in config)."""

    max_steps: int = 1000
    token_budget: int = 16_384  # GUESS: fits a `small` model on a 16 GiB GPU with optimizer-state headroom
    # Micro-batches accumulated per optimizer step. The effective batch is
    # grad_accum_steps × token_budget × world_size, so a large, stable batch can be
    # reached on a memory-bound device without raising token_budget (the per-forward
    # memory peak). One optimizer step (and the LR schedule) advances per window; with
    # the default 1 the trajectory is byte-identical to no accumulation.
    grad_accum_steps: int = 1
    lr_muon: float = 3e-3  # GUESS
    lr_adamw: float = 3e-4  # GUESS
    weight_decay: float = 0.01
    warmup_steps: int = 100  # GUESS
    grad_clip: float = 1.0
    checkpoint_every_min: float = 15.0
    log_every: int = 10
    seed: int = 0
    # Opt-in reproducible CUDA path (default OFF; CPU is already byte-exact from
    # a fixed seed). When True, the GPU forward/embedding is reproducible
    # run-to-run on fixed hardware and training is bit-exact in fp32 — so a
    # deterministic run selects "32-true" on CUDA instead of "bf16-mixed". A
    # deterministic bf16 run is run-to-run stable to ~1e-3 but NOT bit-exact:
    # SDPA bf16/fp16 backward on CUDA has no deterministic implementation
    # upstream. See seed_everything for the global switches this enables.
    deterministic: bool = False
    nan_skip: bool = True
    # Abort if this many consecutive steps are skipped for a non-finite loss/grad
    # — a transient bf16 overflow is recoverable, but a sustained run of skips is
    # divergence and should fail loud rather than burn compute to max_steps.
    max_consecutive_skips: int = 50
    # Observability: a one-line stderr heartbeat every log_every steps, and an
    # optional Weights & Biases mirror (needs `pip install -e ".[tracking]"`).
    # metrics.jsonl and the TensorBoard mirror (runs/{name}/tb, active when the
    # `tensorboard` package is installed) are always on.
    verbose: bool = True
    wandb: bool = False
    wandb_project: str = "pragmatiq"
    # Masking strategy, resolved from the registry by name (rule 8) so configs
    # can swap in a custom @register_masker without forking the trainer.
    masker: str = "pragma"
    p_token: float = 0.15
    p_event: float = 0.10
    p_key: float = 0.10
    p_unk: float = 0.10  # GUESS: fraction of selected → [UNK], excluded from loss
    # PRAGMA+Nemotron variant: weight λ on the text MSE reconstruction term added to
    # the CE loss (loss = CE + λ·MSE). Only active when the model has a text encoder.
    text_loss_weight: float = 1.0
    # Multi-device / multi-node DDP (Fabric). devices: per-node device count or "auto"
    # (all visible GPUs, else 1 CPU process); num_nodes: hosts in the job. The rank
    # sampler shards the data per global rank and the masking seed is offset per rank,
    # so adding ranks trains disjoint slices in lockstep with an independent mask stream.
    devices: int | str = "auto"
    num_nodes: int = 1


def resolve_device_count(devices: int | str, use_cuda: bool) -> int:
    """Resolve a ``devices`` setting to a per-node device count.

    An explicit count is honored whether it arrives as an ``int`` or a numeric
    string (configs loaded from YAML/OmegaConf deliver ``"2"`` as a string); any
    other value (``"auto"``) uses every visible CUDA device, or 1 on CPU.
    """
    if isinstance(devices, bool):  # bool is an int subclass; treat as "unset"
        devices = "auto"
    if isinstance(devices, int):
        return max(1, devices)
    if isinstance(devices, str) and devices.isdigit():
        return max(1, int(devices))
    return torch.cuda.device_count() if use_cuda else 1


def _make_fabric(devices: int | str = "auto", precision: str | None = None,
                 deterministic: bool = False, num_nodes: int = 1):
    try:
        from lightning.fabric import Fabric
    except ImportError as _e:
        from pragmatiq.core.errors import MissingExtraError
        raise MissingExtraError.for_extra("train", "lightning") from _e

    use_cuda = torch.cuda.is_available()
    if precision is None:
        # bf16 backward on CUDA is not bit-exact; a deterministic GPU run trains
        # in fp32 so the gradient is reproducible. CPU is fp32 regardless.
        precision = ("32-true" if (deterministic or not use_cuda) else "bf16-mixed")
    accelerator = "cuda" if use_cuda else "cpu"
    n = resolve_device_count(devices, use_cuda)
    fabric = Fabric(accelerator=accelerator, devices=max(1, n), num_nodes=max(1, num_nodes),
                    precision=precision)  # type: ignore[arg-type]
    fabric.launch()
    return fabric


class PreTrainer:
    """Drives MLM pretraining of a :class:`PragmaModel` with an :class:`MLMHead`."""

    def __init__(
        self,
        model: PragmaModel,
        run: Run,
        config: TrainConfig,
        tokenizer_hash: str,
        masker: MaskingStrategy | None = None,
        fabric: Any = None,
        logger: MetricLogger | None = None,
    ) -> None:
        self.config = config
        self.run = run
        self.tokenizer_hash = tokenizer_hash
        self.fabric = fabric or _make_fabric(
            devices=config.devices, deterministic=config.deterministic, num_nodes=config.num_nodes
        )
        self.masker = masker or get_masker(config.masker)(
            p_token=config.p_token, p_event=config.p_event, p_key=config.p_key, p_unk=config.p_unk
        )
        text_dim = model.config.text_encoder_dim if model.config.text_encoder else 0
        self.head = MLMHead(model.config.dim, text_dim=text_dim)
        self.optimizers, self.opt_names = build_optimizers(
            torch.nn.ModuleList([model, self.head]),
            lr_muon=config.lr_muon, lr_adamw=config.lr_adamw, weight_decay=config.weight_decay,
        )
        self.scheduler = WarmupCosine(self.optimizers, config.warmup_steps, config.max_steps)
        # fabric.setup(module, *optimizers) treats extra positionals as
        # optimizers and never moves them to the device — set up separately.
        self.model = self.fabric.setup(model)
        self.head = self.fabric.setup(self.head)
        self.optimizers = [self.fabric.setup_optimizers(o) for o in self.optimizers]
        # Device-matched masking RNG: torch.rand on CUDA requires a CUDA
        # generator, so this generator lives on fabric.device. That makes it the
        # single source of truth for masking reproducibility on both CPU and GPU
        # (its checkpointed state fully determines the masking stream).
        # Offset the masking seed per DDP rank so ranks (which now train disjoint
        # data slices) also draw independent masking streams. Single-process runs
        # use global_rank 0, so the seed is unchanged and resume stays bit-exact.
        global_rank = int(getattr(self.fabric, "global_rank", 0))
        self.gen = torch.Generator(device=self.fabric.device)
        self.gen.manual_seed(config.seed + global_rank)
        self.step = 0
        self.epoch = 0
        self.logger = logger
        self._tokens_seen = 0
        self._consec_skips = 0
        self._epoch_produced = False

    # ------------------------------------------------------------------ step
    def _zero_graph_loss(self, out: Any) -> torch.Tensor:
        """A graph-connected zero that flows through the SAME model+head graph as a real
        micro-batch, contributing exactly zero gradient.

        Under DDP every rank must call ``fabric.backward`` the same number of times per
        window or the single per-window gradient all-reduce is skipped on some rank and
        the collective deadlocks. When a rank's micro-batch selects nothing to learn from,
        it backwards this instead of skipping. The loss MUST route through the autograd
        graph of the ``out`` produced by ``self.model(...)``: DDP's reducer is armed by the
        forward and waits for a gradient on every forward-touched parameter, so an
        unrelated ``Σp`` loss (which never reaches the forward graph) would hang the
        reduce. Selecting all tokens with all-``-100`` targets makes :func:`mlm_loss`
        return its ``logits.sum()*0`` branch — graph-connected through the head + tied
        embedding + the full model forward, hence the same reducer buckets a real
        micro-batch fills, all with zero gradient.
        """
        n_tok = int(out.token_repr.shape[0])
        sel = torch.arange(n_tok, device=self.fabric.device)
        logits = self.head(out, self.model.embedding_weight, sel)
        targets = torch.full((n_tok,), -100, dtype=torch.int64, device=self.fabric.device)
        loss = mlm_loss(logits, targets)  # all-ignored -> graph-connected zero
        if self.head.text_out is not None and out.text_vecs is not None:
            # Keep the text-head bucket fed too (zero gradient) so DDP doesn't wait on it.
            pred = self.head.reconstruct_text(out, out.text_token_idx)
            loss = loss + pred.sum() * 0.0
        return loss

    def _micro_backward(
        self, batch: PackedBatch, accum: int, agg: dict[str, Any],
        *, sync: bool = True, ddp: bool = False,
    ) -> bool | None:
        """One micro-batch: forward + scaled backward, folding metrics into ``agg``.

        Returns True if it contributed gradient, False on a non-finite loss (skip the
        whole window), or None if nothing was selected to learn from this micro-batch.
        The loss is scaled by ``1/accum`` here; if fewer than ``accum`` micro-batches end
        up contributing, :meth:`_train_step` rescales the accumulated gradient to the mean
        over the contributing ones, so the step matches a single batch of that size.

        ``sync`` is False for the non-final micro-batches of a DDP window: the backward
        runs under ``fabric.no_backward_sync`` so the gradient all-reduce fires only on
        the last (``sync=True``) micro-batch — once per optimizer step, the standard
        Fabric grad-accum pattern. ``ddp`` requests the collectively-symmetric path so
        every rank calls ``fabric.backward`` the same number of times regardless of its
        local data: an empty micro-batch issues a zero-graph backward (rather than
        returning None), AND a non-finite-loss micro-batch ALSO issues a zero-graph
        backward (rather than skipping the backward) before returning the False skip flag.
        The non-finite skip is then decided COLLECTIVELY in :meth:`_train_step`, so a NaN
        on one rank never makes it early-return past the all-reduce the others are blocked
        on (the DDP deadlock the audit follow-up fixes).
        """
        batch = batch.to(self.fabric.device)
        masked = self.masker(batch, self.gen)
        agg["_last_batch"], agg["_last_masked"] = batch, masked  # for the NaN debug dump
        masked_batch = dataclasses.replace(
            batch, value_ids=masked.input_value_ids, feed_text=masked.feed_text
        )
        out = self.model(masked_batch)
        sel = masked.selected_idx
        text_idx = masked.text_loss_idx
        if sel.numel() == 0 and text_idx.numel() == 0:
            if not ddp:
                return None
            # DDP: keep the per-rank backward count symmetric. Backward a zero-graph loss
            # routed through this micro-batch's own forward graph (no gradient
            # contribution) so this rank still hits the all-reduce without orphaning the
            # DDP-armed forward — see _zero_graph_loss.
            self._backward(self._zero_graph_loss(out), sync=sync)
            return None
        logits = self.head(out, self.model.embedding_weight, sel)
        targets = masked.labels[sel]
        ce = mlm_loss(logits, targets)
        loss = ce
        # Nemotron variant: reconstruct masked text tokens' frozen embeddings with MSE.
        text_mse = None
        if self.head.text_out is not None and out.text_vecs is not None:
            pred = self.head.reconstruct_text(out, text_idx)
            text_mse = text_mse_loss(pred, self._text_targets(out, text_idx))
            loss = ce + self.config.text_loss_weight * text_mse
        if self.config.nan_skip and not torch.isfinite(loss):
            if not ddp:
                return False
            # DDP: a non-finite loss on this rank must NOT skip the backward — every
            # rank has to issue the same number of backwards so the single per-window
            # gradient all-reduce fires symmetrically (else the others deadlock on it).
            # Backward a zero-graph loss (no gradient) through this micro-batch's own
            # forward graph and report the skip via the False flag; _train_step makes the
            # actual skip decision COLLECTIVELY so all ranks branch together.
            self._backward(self._zero_graph_loss(out), sync=sync)
            return False
        self._backward(loss / accum, sync=sync)
        with torch.no_grad():
            agg["loss_sum"] += loss.item()
            agg["contributing"] += 1
            if targets.numel():
                agg["acc_correct"] += float((logits.argmax(-1) == targets).sum().item())
                agg["acc_total"] += int(targets.numel())
                mtype = masked.mask_type[sel]
                for code, name in TYPE_NAMES.items():
                    m = mtype == code
                    n = int(m.sum())
                    if n:
                        agg[f"loss_{name}_sum"] = agg.get(f"loss_{name}_sum", 0.0) + \
                            mlm_loss(logits[m], targets[m]).item() * n
                        agg[f"loss_{name}_n"] = agg.get(f"loss_{name}_n", 0) + n
            if text_mse is not None:
                agg["text_mse_sum"] += text_mse.item()
                agg["text_mse_n"] += 1
        self._tokens_seen += batch.n_tokens
        return True

    def _backward(self, loss: torch.Tensor, *, sync: bool) -> None:
        """``fabric.backward(loss)``; when ``sync`` is False defer the DDP gradient
        all-reduce via ``no_backward_sync`` on both Fabric-wrapped modules (model + head
        both receive gradients), so the reduce fires only on the final window backward."""
        if sync:
            self.fabric.backward(loss)
            return
        with self.fabric.no_backward_sync(self.model), self.fabric.no_backward_sync(self.head):
            self.fabric.backward(loss)

    def _train_step(self, micro_batches: list[PackedBatch]) -> dict[str, float] | None:
        accum = len(micro_batches)
        world_size = int(getattr(self.fabric, "world_size", 1))
        ddp = world_size > 1
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        agg: dict[str, Any] = {"loss_sum": 0.0, "contributing": 0, "acc_correct": 0.0,
                               "acc_total": 0, "text_mse_sum": 0.0, "text_mse_n": 0}
        local_skip = False  # this rank saw a non-finite micro-batch loss
        for i, mb in enumerate(micro_batches):
            # Under DDP defer the gradient all-reduce to the LAST micro-batch of the
            # window (no_backward_sync on the rest) so it fires once per optimizer step;
            # world_size==1 always syncs, leaving the single-process path byte-identical.
            sync = (not ddp) or (i == accum - 1)
            result = self._micro_backward(mb, accum, agg, sync=sync, ddp=ddp)
            if result is False:  # non-finite loss → dump + skip the whole window
                if not ddp:
                    # Single-process: early-return is safe (no collective to deadlock on)
                    # and keeps the byte-identical behavior the single-process NaN-skip
                    # tests pin down.
                    self._dump_debug(agg["_last_batch"], agg["_last_masked"])
                    for opt in self.optimizers:
                        opt.zero_grad(set_to_none=True)
                    return {"loss": float("nan"), "skipped": 1.0}
                # DDP: this rank backwarded a zero-graph loss (in _micro_backward) so the
                # per-window all-reduce still fires symmetrically. Record the skip (and
                # snapshot the offending micro-batch for the debug dump, since the loop
                # keeps going and `_last_batch` would otherwise advance past it) and keep
                # iterating — the skip decision is made COLLECTIVELY below so every rank
                # branches together (no rank early-returns past a pending collective).
                local_skip = True
                agg.setdefault("_skip_batch", agg["_last_batch"])
                agg.setdefault("_skip_masked", agg["_last_masked"])
        # A finite loss can still produce non-finite grads (e.g. bf16 backward overflow);
        # treat that like a NaN loss (dump + skip). Computed locally here, then folded into
        # the same collective skip decision so a non-finite grad on ONE rank makes ALL
        # ranks skip together. The per-window gradient all-reduce has already fired (the
        # last backward synced), so grads are the post-reduce values on every rank.
        local_grads_nonfinite = self.config.nan_skip and not self._grads_finite()
        local_contributing = agg["contributing"]
        if ddp:
            # ONE combined per-window collective every rank ALWAYS reaches: sum the per-rank
            # `contributing` count AND the skip flags so any rank's non-finite loss/grad
            # propagates to all ranks (sum > 0 ⇒ "some rank flagged" ⇒ everyone skips). All
            # three quantities ride one all-reduce to keep it to a single collective.
            packed = self.fabric.all_reduce(
                torch.tensor(
                    [float(local_contributing),
                     float(local_skip),
                     float(local_grads_nonfinite)],
                    device=self.fabric.device,
                ),
                reduce_op="sum",
            )
            global_contributing = int(round(float(packed[0])))
            global_skip = float(packed[1]) > 0.0
            global_grads_nonfinite = float(packed[2]) > 0.0
        else:
            global_contributing = local_contributing
            global_skip = local_skip  # always False here (single-process took the early return)
            global_grads_nonfinite = local_grads_nonfinite
        # Branch the SAME way on every rank: any rank that saw a non-finite micro-batch
        # loss makes all ranks skip the step (dump + zero grads + consistent skip return),
        # so replicas stay in sync and no rank steps while another skipped.
        if global_skip:
            # Dump the offending micro-batch if THIS rank saw the non-finite loss; a rank
            # that only skips because a PEER flagged it dumps its current last batch (best
            # effort — its own data was finite).
            self._dump_debug(
                agg.get("_skip_batch", agg["_last_batch"]),
                agg.get("_skip_masked", agg["_last_masked"]),
            )
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)
            return {"loss": float("nan"), "skipped": 1.0}
        if global_contributing == 0:
            return None
        # Each micro-batch backward scaled its loss by 1/accum. DDP additionally
        # mean-reduces gradients across `world_size` ranks, so after the window each
        # parameter's grad is Σ(grad_i over all ranks) / (world_size·accum). We want the
        # mean over the globally-contributing micro-batches, Σ(grad_i) / global_contributing
        # — matching a single-process step over the same total data — so the rescale factor
        # is (world_size·accum)/global_contributing, identical on every rank. With
        # world_size==1 this is accum/contributing, the byte-identical single-process path;
        # and when every micro-batch on every rank contributes it is 1.0 (a no-op).
        rescale = (world_size * accum) / global_contributing
        if rescale != 1.0:
            self._scale_grads(rescale)
        if local_contributing == 0:
            # This rank's grad is purely zero-graph backwards (and the all-reduced share
            # of other ranks); the loss metric below would divide by zero. Report the
            # global average loss is unavailable here, so fall back to 0.0 for this rank.
            avg_loss = 0.0
        else:
            avg_loss = agg["loss_sum"] / local_contributing
        # Non-finite grad skip (decided collectively above): all ranks branch together.
        if global_grads_nonfinite:
            self._dump_debug(agg["_last_batch"], agg["_last_masked"])
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)
            return {"loss": avg_loss, "skipped": 1.0}
        # Clip each optimizer's params to grad_clip (Muon over hidden weights +
        # AdamW over embeddings/norms/biases). Clipping is PER-OPTIMIZER — the
        # `module` arg is ignored on the default strategy; clip_gradients selects
        # params by optimizer — so the effective global bound is
        # ~sqrt(n_opt)*grad_clip and the reported grad_norm is the pre-clip quadrature.
        gn_muon = self.fabric.clip_gradients(self.model, self.optimizers[0], max_norm=self.config.grad_clip)
        gn_rest = [self.fabric.clip_gradients(self.head, opt, max_norm=self.config.grad_clip)
                   for opt in self.optimizers[1:]]
        gnorm = sum(float(g) ** 2 for g in [gn_muon, *gn_rest]) ** 0.5
        # Set the LR from the global step BEFORE the update (warmup-correct and
        # skip/NaN-proof); fit() advances self.step after this returns.
        lr_factor = self.scheduler.apply(self.step)
        for opt in self.optimizers:
            opt.step()

        metrics = {"loss": avg_loss, "grad_norm": gnorm, "lr_factor": lr_factor}
        if agg["acc_total"]:
            metrics["mlm_acc"] = agg["acc_correct"] / agg["acc_total"]
            for name in TYPE_NAMES.values():
                if agg.get(f"loss_{name}_n"):
                    metrics[f"loss_{name}"] = agg[f"loss_{name}_sum"] / agg[f"loss_{name}_n"]
        if agg["text_mse_n"]:
            metrics["loss_text_mse"] = agg["text_mse_sum"] / agg["text_mse_n"]
        return metrics

    def _text_targets(self, out: Any, text_idx: torch.Tensor) -> torch.Tensor:
        """Frozen text embeddings (the MSE targets) for the masked text token positions."""
        n_text = int(out.text_token_idx.numel())
        row = torch.full((out.token_repr.shape[0],), -1, dtype=torch.long, device=text_idx.device)
        row[out.text_token_idx] = torch.arange(n_text, device=text_idx.device)
        return out.text_vecs[row[text_idx]]

    def _scale_grads(self, factor: float) -> None:
        """Multiply every accumulated grad in place by ``factor`` (corrects the window
        average when fewer micro-batches contributed than the backward divisor)."""
        for opt in self.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.mul_(factor)

    def _grads_finite(self) -> bool:
        """True iff every trainable grad is finite (used by the NaN-skip guard)."""
        for opt in self.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    g = p.grad
                    if g is not None and not torch.isfinite(g).all():
                        return False
        return True

    def _dump_debug(self, batch: PackedBatch, masked: Any) -> None:
        dbg = self.run.dir / "debug"
        dbg.mkdir(exist_ok=True)
        torch.save({"user_ids": batch.user_ids, "key_ids": batch.key_ids.cpu(),
                    "value_ids": batch.value_ids.cpu(), "step": self.step},
                   dbg / f"nan_step{self.step}.pt")

    # ------------------------------------------------------------------ fit
    def fit(self, loader: ShardDataLoader, resume: str | None = None,
            max_steps: int | None = None) -> Run:
        """Train until ``max_steps`` (default ``config.max_steps``); checkpoint
        periodically and at the end.

        ``max_steps`` only bounds the loop — the LR schedule horizon stays
        ``config.max_steps`` — so stopping early then resuming reproduces the
        uninterrupted trajectory exactly (the resume contract).
        """
        stop_at = self.config.max_steps if max_steps is None else max_steps
        resumed = False
        ckpt_path = self.run.last_checkpoint()
        if resume == "auto" and ckpt_path is not None:
            self.load_checkpoint(ckpt_path, loader)
            resumed = True  # sampler already positioned; do NOT reset it
        # Shard the batch stream for the current world (after any resume restored
        # the sampler position). world_size=1 → no-op, single-process path intact.
        loader.sampler.set_replica_info(
            int(getattr(self.fabric, "world_size", 1)),
            int(getattr(self.fabric, "global_rank", 0)),
        )
        last_ckpt = time.time()
        t0 = time.time()
        # Rates must count only THIS fit() call: after a resume, step and
        # _tokens_seen carry the checkpointed totals while t0 restarts.
        start_step = self.step
        tokens0 = self._tokens_seen
        if not resumed:
            loader.sampler.set_epoch(self.epoch)
        data_iter = iter(loader)
        self._epoch_produced = False
        accum = max(1, self.config.grad_accum_steps)

        def _next_window() -> list[PackedBatch]:
            # Gather `accum` micro-batches for one optimizer step, rolling the epoch
            # when the sampler is exhausted mid-window (gradients still accumulate
            # across the boundary; checkpoints land only between whole windows).
            nonlocal data_iter
            window: list[PackedBatch] = []
            while len(window) < accum:
                try:
                    window.append(next(data_iter))
                    self._epoch_produced = True
                except StopIteration:
                    # A loader that yields nothing for a whole epoch would otherwise
                    # spin here forever re-iterating an empty stream — fail with an
                    # actionable message instead.
                    if not self._epoch_produced:
                        raise RuntimeError(
                            "the data loader produced no batches for a full epoch; verify "
                            "the shard directory is non-empty and TrainConfig.token_budget "
                            f"(currently {self.config.token_budget}) admits at least one "
                            "user per batch"
                        ) from None
                    self.epoch += 1
                    loader.sampler.set_epoch(self.epoch)
                    data_iter = iter(loader)
                    self._epoch_produced = False
            return window

        while self.step < stop_at:
            window = _next_window()
            try:
                metrics = self._train_step(window)
            except torch.cuda.OutOfMemoryError:
                # Clear transient fragmentation and retry once; if it still does not
                # fit, fail with an actionable message rather than a raw CUDA error
                # (training-time recovery is a token_budget / grad_accum choice).
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    metrics = self._train_step(window)
                except torch.cuda.OutOfMemoryError as e2:
                    peak = max(b.n_tokens for b in window)
                    raise RuntimeError(
                        f"CUDA OOM on a {peak}-token micro-batch at step {self.step}; lower "
                        f"TrainConfig.token_budget (currently {self.config.token_budget}) "
                        f"or raise grad_accum_steps, or use a larger GPU."
                    ) from e2
                log.warning("CUDA OOM at step %d; cleared cache and retried successfully", self.step)
            if metrics is None:
                continue
            self.step += 1
            is_zero = getattr(self.fabric, "is_global_zero", True)
            if metrics.get("skipped"):
                # Surface every skip (not gated by log_every) and abort if skips
                # run away — a sustained non-finite streak is divergence.
                self._consec_skips += 1
                if self.logger is not None and is_zero:
                    self.logger.log(self.step, metrics)
                if self.config.verbose and is_zero:
                    print(f"step {self.step}: non-finite loss/grad — batch dumped, step skipped "
                          f"({self._consec_skips} in a row)", file=sys.stderr, flush=True)
                if self._consec_skips > self.config.max_consecutive_skips:
                    raise RuntimeError(
                        f"{self._consec_skips} consecutive non-finite steps at step {self.step} "
                        f"(> max_consecutive_skips={self.config.max_consecutive_skips}); aborting as "
                        f"divergence. Lower the learning rate or inspect {self.run.dir}/debug/."
                    )
                continue
            self._consec_skips = 0
            if self.step % self.config.log_every == 0 or self.step == 1:
                elapsed = max(time.time() - t0, 1e-6)
                # Under DDP each rank processes a disjoint slice; rank 0 reports
                # its local throughput scaled by world_size as the aggregate.
                ws = int(getattr(self.fabric, "world_size", 1))
                metrics["tokens_per_sec"] = (self._tokens_seen - tokens0) * ws / elapsed
                if torch.cuda.is_available():
                    metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                if self.logger is not None and is_zero:
                    self.logger.log(self.step, metrics)
                if self.config.verbose and is_zero:
                    steps_per_sec = (self.step - start_step) / elapsed
                    eta_min = (stop_at - self.step) / max(steps_per_sec, 1e-9) / 60.0
                    print(
                        f"step {self.step}/{stop_at}  loss {metrics['loss']:.4f}  "
                        f"mlm_acc {metrics.get('mlm_acc', 0.0):.3f}  "
                        f"{metrics['tokens_per_sec']:,.0f} tok/s  eta {eta_min:.1f}m",
                        file=sys.stderr, flush=True,
                    )
            if (time.time() - last_ckpt) / 60.0 >= self.config.checkpoint_every_min:
                self.save_checkpoint(loader, "last.pt")
                last_ckpt = time.time()
        self.save_checkpoint(loader, "last.pt")
        return self.run

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self, loader: ShardDataLoader) -> dict[str, Any]:
        """Assemble the full resumable training state (global rule 3)."""
        return {
            "format": CKPT_FORMAT,
            "step": self.step,
            "epoch": self.epoch,
            "tokens_seen": self._tokens_seen,
            "model": self.model.state_dict(),
            "head": self.head.state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "opt_names": self.opt_names,
            "scheduler": self.scheduler.state_dict(),
            "sampler": loader.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": _pack_numpy_state(np.random.get_state()),
                "masking_gen": self.gen.get_state(),
                "masking_gen_device": self.gen.device.type,
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "tokenizer_hash": self.tokenizer_hash,
            "model_config": dataclasses.asdict(self.model.config),
            "train_config": dataclasses.asdict(self.config),
        }

    def save_checkpoint(self, loader: ShardDataLoader, filename: str) -> Path:
        """Write a full checkpoint atomically to ``checkpoints/{filename}``."""
        path = self.run.checkpoints / filename
        tmp = path.with_suffix(".tmp")
        # fabric.save writes the temp file on the global-zero rank only; the
        # rename must run there too (other ranks would race / hit the moved file),
        # then a barrier holds all ranks until the rename is visible.
        self.fabric.save(tmp, self.state_dict(loader))
        if getattr(self.fabric, "is_global_zero", True):
            tmp.replace(path)
        barrier = getattr(self.fabric, "barrier", None)
        if callable(barrier):
            barrier()
        return path

    def load_checkpoint(self, path: str | Path, loader: ShardDataLoader) -> None:
        """Restore model/optimizers/scheduler/sampler/RNG from a checkpoint."""
        ckpt = self.fabric.load(path)
        if ckpt.get("format") != CKPT_FORMAT:
            raise ValueError(
                f"unsupported checkpoint format {ckpt.get('format')!r}; expected {CKPT_FORMAT}"
            )
        if ckpt.get("tokenizer_hash") != self.tokenizer_hash:
            raise ValueError(
                f"tokenizer hash mismatch: checkpoint {ckpt.get('tokenizer_hash')!r} != "
                f"current {self.tokenizer_hash!r}. Refusing to resume with a different tokenizer."
            )
        _require_matching_config("model_config", ckpt.get("model_config"), dataclasses.asdict(self.model.config))
        _require_matching_config(
            "train_config",
            ckpt.get("train_config"),
            dataclasses.asdict(self.config),
            ignored=_CHECKPOINT_OPERATIONAL_KEYS,
        )
        self.model.load_state_dict(ckpt["model"])
        self.head.load_state_dict(ckpt["head"])
        for opt, st in zip(self.optimizers, ckpt["optimizers"]):
            opt.load_state_dict(st)
        self.scheduler.load_state_dict(ckpt["scheduler"])
        # Resume can extend the run (a larger max_steps stretches the cosine
        # horizon); pin the schedule to the current config and log when it moves.
        # Same-horizon resume leaves total_steps untouched, so bit-exact resume holds.
        if self.config.max_steps != self.scheduler.total_steps:
            log.info("resume extends schedule horizon: %d -> %d",
                     self.scheduler.total_steps, self.config.max_steps)
            self.scheduler.total_steps = self.config.max_steps
        loader.load_state_dict(ckpt["sampler"])
        self.step = ckpt["step"]
        self.epoch = ckpt["epoch"]
        self._tokens_seen = ckpt["tokens_seen"]
        # Drop append-only metric rows past the resumed step: logging cadence
        # (log_every) and checkpoint cadence (every N min) are independent, so a
        # mid-interval crash can leave logged-but-uncheckpointed rows that would
        # otherwise duplicate and break monotonicity after resume. Only the
        # global-zero rank owns metrics.jsonl, so only it rewrites the file.
        if self.logger is not None and getattr(self.fabric, "is_global_zero", True):
            self.logger.truncate_after(self.step)
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(_unpack_numpy_state(rng["numpy"]))
        # The masking generator is device-typed; its raw state is not portable
        # across device PRNG algorithms (CPU MT19937 vs CUDA Philox). Restore it
        # only on a matching device, else deterministically re-seed from this step.
        saved_dev = rng.get("masking_gen_device")
        if saved_dev is None or saved_dev == self.gen.device.type:
            self.gen.set_state(rng["masking_gen"])
        else:
            reseed = self.config.seed + int(getattr(self.fabric, "global_rank", 0)) + self.step
            self.gen.manual_seed(reseed)
            log.warning(
                "masking RNG was saved on %s but resuming on %s; PRNG state is not "
                "portable across devices — re-seeding the masking stream from step %d.",
                saved_dev, self.gen.device.type, self.step,
            )
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
