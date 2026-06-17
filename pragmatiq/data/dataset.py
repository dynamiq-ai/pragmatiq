"""Shard dataset + dynamic token-budget batch sampler (Phase 3).

:class:`DynamicBatchSampler` greedily packs users into batches under a token
budget, grouping by event-count band so batches are length-homogeneous. It is
fully resumable: ``state_dict()`` / ``load_state_dict()`` capture the shuffle
RNG, the epoch, and how many batches have been consumed, so a killed run
resumes the exact same batch stream (used by the gate-5 resume test).

:class:`ShardDataset` lazily materializes :class:`TokenizedRecord` objects from
the parquet shards by ``(band, shard, row)`` with a small LRU shard cache.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .collate import PackedBatch, VarlenCollator
from .sharding import UserIndex, record_from_row
from .tokenizer import TokenizedRecord


class ShardDataset:
    """Random-access view over tokenized records stored in parquet shards."""

    def __init__(self, shard_dir: str | Path, cache_shards: int = 4) -> None:
        self.dir = Path(shard_dir)
        self.index = UserIndex(self.dir)
        self._cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._cache_n = cache_shards

    def __len__(self) -> int:
        return len(self.index)

    @property
    def user_ids(self) -> list[str]:
        return self.index.order

    def _shard_table(self, band: int, shard: int) -> Any:
        key = (band, shard)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.dir / "shards" / f"band{band}_shard{shard:05d}.parquet"
        table = pq.read_table(path)
        self._cache[key] = table
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return table

    def get(self, user_id: str) -> TokenizedRecord:
        """Materialize one user's tokenized record."""
        m = self.index.meta(user_id)
        table = self._shard_table(m.band, m.shard)
        row = table.slice(m.row, 1).to_pylist()[0]
        return record_from_row(row)

    def get_many(self, user_ids: list[str]) -> list[TokenizedRecord]:
        """Materialize several records, grouped by shard so each shard is
        converted to Python once (``table.take`` + a single ``to_pylist``)
        instead of one ``slice(row,1).to_pylist()`` per user — byte-identical to
        per-user ``get`` but ~100x faster on the embed / baseline tail where many
        users share a shard. Output order matches ``user_ids``."""
        from collections import defaultdict

        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for out_pos, uid in enumerate(user_ids):
            m = self.index.meta(uid)
            groups[(m.band, m.shard)].append((out_pos, m.row))
        out: list[Any] = [None] * len(user_ids)
        for (band, shard), items in groups.items():
            table = self._shard_table(band, shard)
            rows = table.take([r for _, r in items]).to_pylist()
            for (out_pos, _), row in zip(items, rows):
                out[out_pos] = record_from_row(row)
        return out

    def close(self) -> None:
        self.index.close()


class DynamicBatchSampler:
    """Greedy token-budget batches over user indices, resumable and shuffled.

    Each yielded batch is a list of integer user indices whose summed token
    count stays under ``token_budget`` (a single user above budget forms its own
    batch). Users are bucketed by band and shuffled within band each epoch so a
    batch holds similar-length histories (cheap packing, low waste).
    """

    def __init__(
        self,
        index: UserIndex,
        token_budget: int = 16_384,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        subset: list[int] | None = None,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        # Budget on event + profile tokens so collated batches honor the budget.
        self.n_tokens = np.asarray(index.n_tokens, dtype=np.int64) + np.asarray(
            index.n_prof_tokens, dtype=np.int64
        )
        self.bands = np.asarray(index.bands, dtype=np.int16)
        self.token_budget = int(token_budget)
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        # `subset` restricts batching to a set of users by their position in the
        # index (e.g. embedding only a labeled cohort) without changing the
        # position->user_id mapping the loader relies on. None batches everyone.
        if subset is None:
            self._keep: np.ndarray | None = None
        else:
            keep = np.zeros(self.bands.shape[0], dtype=bool)
            keep[np.asarray(subset, dtype=np.int64)] = True
            self._keep = keep
        # Data-parallel sharding: under DDP each rank trains a disjoint slice of
        # the batches so the all-reduced gradient covers the whole epoch once.
        # num_replicas=1 (the default) is the single-process / CPU path, unchanged.
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self._epoch = 0
        self._consumed = 0  # batches consumed in the current epoch (resume offset)
        self._batches: list[list[int]] | None = None

    def set_replica_info(self, num_replicas: int, rank: int) -> None:
        """Configure data-parallel sharding (rank ``rank`` of ``num_replicas``).

        Called by the trainer once the Fabric/world size is known. Invalidates the
        current plan so the next epoch is re-sharded for this rank.
        """
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self._batches = None

    # ------------------------------------------------------------------ planning
    def _plan(self, epoch: int) -> list[list[int]]:
        rng = np.random.default_rng((self.seed, epoch))
        order: list[int] = []
        for b in np.unique(self.bands):
            idx = np.nonzero(self.bands == b)[0]
            if self._keep is not None:
                idx = idx[self._keep[idx]]
            if self.shuffle:
                rng.shuffle(idx)
            order.extend(idx.tolist())
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_tok = 0
        budget = self.token_budget
        for i in order:
            t = int(self.n_tokens[i]) + 1  # +1 for the [USR] slot
            if cur and cur_tok + t > budget:
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(i)
            cur_tok += t
            if t >= budget:  # oversized user already exceeds budget alone
                batches.append(cur)
                cur, cur_tok = [], 0
        if cur and not self.drop_last:
            batches.append(cur)
        if self.num_replicas > 1 and batches:
            # Pad (by cycling) to a multiple of the world size so every rank
            # yields the same number of steps — unequal step counts deadlock the
            # DDP gradient all-reduce — then take this rank's disjoint stride.
            pad = (-len(batches)) % self.num_replicas
            if pad:
                batches = batches + [batches[i % len(batches)] for i in range(pad)]
            batches = batches[self.rank :: self.num_replicas]
        return batches

    def set_epoch(self, epoch: int) -> None:
        """Start a fresh epoch (re-plans batches, resets the consumed counter)."""
        self._epoch = epoch
        self._consumed = 0
        self._batches = self._plan(epoch)

    def __len__(self) -> int:
        if self._batches is None:
            self._batches = self._plan(self._epoch)
        return len(self._batches)

    def __iter__(self) -> Iterator[list[int]]:
        if self._batches is None:
            self._batches = self._plan(self._epoch)
        # resume from the consumed offset, then advance the counter as we go
        start = self._consumed
        for bi in range(start, len(self._batches)):
            self._consumed = bi + 1
            yield self._batches[bi]
        # epoch finished: advance to the next and reset offset
        self._epoch += 1
        self._consumed = 0
        self._batches = None

    # ------------------------------------------------------------------ resume
    def state_dict(self) -> dict[str, Any]:
        """Serializable sampler position (epoch, consumed batches, config)."""
        return {
            "epoch": self._epoch,
            "consumed": self._consumed,
            "seed": self.seed,
            "token_budget": self.token_budget,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sampler position so iteration resumes at the same batch."""
        self.seed = state["seed"]
        self.token_budget = int(state["token_budget"])
        self.shuffle = state["shuffle"]
        self.drop_last = state["drop_last"]
        self.num_replicas = max(1, int(state.get("num_replicas", 1)))
        self.rank = int(state.get("rank", 0))
        self._epoch = int(state["epoch"])
        self._consumed = int(state["consumed"])
        self._batches = self._plan(self._epoch)


class ShardDataLoader:
    """Iterates (sampler → dataset → collator) yielding :class:`PackedBatch`.

    A lightweight loader (not ``torch.utils.data.DataLoader``) so the sampler's
    resumable state and the varlen collation stay first-class. Single-process by
    default; the collator is stateless, so wrapping in a worker pool later is
    safe.
    """

    def __init__(
        self,
        dataset: ShardDataset,
        sampler: DynamicBatchSampler,
        collator: VarlenCollator | None = None,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.collator = collator or VarlenCollator()
        self._order = dataset.user_ids

    def __iter__(self) -> Iterator[PackedBatch]:
        for batch_idx in self.sampler:
            uids = [self._order[i] for i in batch_idx]
            records = self.dataset.get_many(uids)
            yield self.collator(records)

    def __len__(self) -> int:
        return len(self.sampler)

    def state_dict(self) -> dict[str, Any]:
        """Sampler resume state (dataset is stateless)."""
        return self.sampler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.sampler.load_state_dict(state)
