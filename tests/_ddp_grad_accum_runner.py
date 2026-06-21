"""Subprocess entry point for the DDP × grad-accumulation equivalence test.

Run as a standalone script so Lightning Fabric can spawn its own gloo worker
processes (``Fabric(devices=N).launch()`` re-executes this module per rank)
without re-entering pytest — the same harness pattern as
``tests/_ddp_finetune_runner.py``.

It drives a SINGLE :meth:`PreTrainer._train_step` over a fixed, deterministic
window of micro-batches and prints the resulting model+head parameters (as a flat
float list under a stable key order) so the test can compare:

- ``mode=ref`` (run with ``devices=1``): one window over ALL ``N`` micro-batches —
  the single-process grad-accum reference.
- ``mode=ddp`` (run with ``devices=2``): the SAME ``N`` micro-batches sharded
  ``list[rank::world]`` across the gloo ranks, one window of ``N/world`` per rank.

With the deterministic content-keyed masker below, both paths mask each global
micro-batch identically and the DDP run's all-reduced + globally-rescaled step must
land on the same parameters as the reference (the equivalence the fix guarantees),
without deadlocking. One designated micro-batch selects NOTHING, so at least one
rank sees an empty micro-batch — the variable-``contributing`` desync case.

Each rank writes its result to ``<out_dir>/result_rank{r}.json`` (Fabric's gloo
launcher does not reliably pipe every rank's stdout back, so a file per rank lets
the test read BOTH ranks' parameters and prove they stayed in sync).

Usage:
    python -m tests._ddp_grad_accum_runner <tok_dir> <out_dir> <mode> <devices>
"""

from __future__ import annotations

import json
import sys

import torch

from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
from pragmatiq.data.tokenizer import MASK, PragmaTokenizer
from pragmatiq.experiments.run import Run
from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel
from pragmatiq.training.masking import T_TOKEN, MaskedBatch
from pragmatiq.training.pretrainer import PreTrainer, TrainConfig, seed_everything

# A small, fixed number of micro-batches (even, so it shards evenly over 2 ranks).
N_MICRO = 4
# The window index (in the global, pre-shard order) whose micro-batch is forced to
# select nothing — the empty micro-batch that exercises the variable-contributing path.
EMPTY_AT = 2


class _DeterministicMasker:
    """Masks the same positions for the same batch content regardless of rank/RNG, so
    the reference and the per-rank DDP forward agree token-for-token on every global
    micro-batch. Position ``EMPTY_AT`` in the global order selects nothing (the empty
    micro-batch); the masker is told each batch's global index at construction time."""

    def __init__(self, empty_global_indices: set[int]) -> None:
        self._empty = empty_global_indices
        self._idx = 0  # advances per __call__, in the order micro-batches are fed

    def __call__(self, batch, generator=None):  # noqa: ANN001
        gidx = self._order[self._idx]
        self._idx += 1
        t = int(batch.key_ids.numel())
        if gidx in self._empty:
            return MaskedBatch(
                input_value_ids=batch.value_ids.clone(),
                labels=torch.full((t,), -100, dtype=torch.int64),
                mask_type=torch.full((t,), -1, dtype=torch.int8),
                selected_idx=torch.zeros(0, dtype=torch.long),
            )
        sel = torch.arange(0, t, 5, dtype=torch.long)
        labels = torch.full((t,), -100, dtype=torch.int64)
        labels[sel] = batch.value_ids[sel]
        ivi = batch.value_ids.clone()
        ivi[sel] = MASK
        mtype = torch.full((t,), -1, dtype=torch.int8)
        mtype[sel] = T_TOKEN
        return MaskedBatch(input_value_ids=ivi, labels=labels, mask_type=mtype, selected_idx=sel)


def _flat_params(trainer: PreTrainer) -> list[float]:
    """Model + head parameters as a flat float list under a stable key order."""
    vals: list[float] = []
    for _, p in sorted(trainer.model.named_parameters()):
        vals.extend(p.detach().reshape(-1).tolist())
    for _, p in sorted(trainer.head.named_parameters()):
        vals.extend(p.detach().reshape(-1).tolist())
    return vals


def main() -> int:
    tok_dir, out_dir, mode, devices_s = sys.argv[1:5]
    devices = int(devices_s)

    tok = PragmaTokenizer.load(tok_dir + "/tokenizer")
    # Identical init in every process (CI-enforced byte-exact from the seed), dropout off
    # so the only randomness left is the (content-keyed, deterministic) masker.
    seed_everything(0)
    model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size, overrides={"dropout": 0.0}))

    # Gather the global, pre-shard micro-batch order from a single-process sampler so the
    # SAME N batches define both the reference window and the sharded DDP windows.
    ds = ShardDataset(tok_dir)
    base_sampler = DynamicBatchSampler(ds.index, token_budget=4096, seed=0, num_replicas=1, rank=0)
    base_sampler.set_epoch(0)
    base_loader = ShardDataLoader(ds, base_sampler)
    all_batches = []
    for b in base_loader:
        all_batches.append(b)
        if len(all_batches) >= N_MICRO:
            break
    if len(all_batches) < N_MICRO:
        # Cycle to reach N_MICRO so the shape is fixed regardless of dataset size.
        all_batches = (all_batches * ((N_MICRO // max(1, len(all_batches))) + 1))[:N_MICRO]

    run = Run.create(f"ddpacc_{mode}", {}, 0, tok.content_hash, tok_dir + "/../runs",
                     tokenizer_src=tok_dir + "/tokenizer")
    cfg = TrainConfig(max_steps=10, token_budget=4096, warmup_steps=0, seed=0,
                      grad_accum_steps=N_MICRO, checkpoint_every_min=1000.0, devices=devices)
    masker = _DeterministicMasker({EMPTY_AT})
    trainer = PreTrainer(model, run, cfg, tok.content_hash, masker=masker)

    world = int(getattr(trainer.fabric, "world_size", 1))
    rank = int(getattr(trainer.fabric, "global_rank", 0))

    if mode == "ref":
        window = all_batches
        global_order = list(range(N_MICRO))  # global indices, in feed order
    else:
        # Shard the global order across ranks; record each fed batch's GLOBAL index so the
        # masker still treats EMPTY_AT as empty on whichever rank owns it.
        global_order = list(range(rank, N_MICRO, world))
        window = [all_batches[g] for g in global_order]
    masker._order = global_order
    # How many of THIS rank's micro-batches are empty (select nothing) — the test asserts
    # the DDP run has a rank with > 0 empties so the variable-`contributing` desync path
    # (the bug) is genuinely exercised, not skipped.
    n_empty = sum(1 for g in global_order if g == EMPTY_AT)

    trainer._train_step(window)

    payload = {"rank": rank, "world": world, "n_empty": n_empty, "n_micro": len(window),
               "params": _flat_params(trainer)}
    ds.close()
    out_path = f"{out_dir}/result_{mode}_rank{rank}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)
    # Also emit on stdout (the global-zero rank, at least, is piped back) as a liveness
    # signal; the test reads the per-rank files for the actual parameter comparison.
    print(f"RESULT rank{rank} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
