"""Batch embedding.

``BatchEmbedder`` streams a tokenized shard directory through a trained model
and writes ``embeddings.parquet`` (user_id + embedding vector), reporting
throughput in users/sec. It reuses the resumable shard dataloader, so it scales
to the full book and runs on CPU or GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.env import inference_context, resolve_device
from ..core.progress import progress
from ..data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
from ..models.pragmatiq import PragmaModel


class BatchEmbedder:
    """Embed every user in a shard directory to a parquet of user vectors.

    ``device="auto"`` picks CUDA when available; ``precision`` selects the CUDA
    autocast dtype (``auto`` = bf16 on CUDA, fp32 on CPU). Batches are collated
    on a prefetch thread and pinned so the GPU never waits on the host.
    """

    def __init__(self, model: PragmaModel, device: str = "auto", token_budget: int = 16_384,
                 precision: str = "auto", prefetch: int = 2) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.token_budget = token_budget
        self.precision = precision
        self.prefetch = prefetch

    def embed_to_parquet(self, shard_dir: str | Path, out_path: str | Path) -> dict[str, Any]:
        """Write ``out_path`` (user_id, embedding) and return throughput stats."""
        ds = ShardDataset(shard_dir)
        sampler = DynamicBatchSampler(ds.index, token_budget=self.token_budget, shuffle=False)
        sampler.set_epoch(0)
        is_cuda = self.device.startswith("cuda")
        loader = ShardDataLoader(ds, sampler, prefetch=self.prefetch, pin_memory=is_cuda)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Stream the output: flush a row group every ``chunk`` users instead of
        # holding every vector in memory, so a full-book embed does not grow
        # unbounded. An empty book still writes a valid file with this schema.
        schema = pa.schema([pa.field("user_id", pa.string()),
                            pa.field("embedding", pa.list_(pa.float32()))])
        chunk = 20_000
        buf_uids: list[str] = []
        buf_vecs: list[np.ndarray] = []
        n_users = 0
        writer = pq.ParquetWriter(out_path, schema, compression="zstd")

        def _flush() -> None:
            if not buf_uids:
                return
            mat = np.concatenate(buf_vecs)
            writer.write_table(pa.table(
                {"user_id": buf_uids,
                 "embedding": pa.array([row.tolist() for row in mat], type=pa.list_(pa.float32()))},
                schema=schema))
            buf_uids.clear()
            buf_vecs.clear()

        t0 = time.time()
        try:
            for batch in progress(loader, total=len(loader), desc="embed", unit="batch"):
                batch = batch.to(self.device, non_blocking=is_cuda)
                with inference_context(self.device, self.precision):
                    z = self.model.embed_users(batch).float().cpu().numpy()
                buf_uids.extend(batch.user_ids)
                buf_vecs.append(z)
                n_users += len(batch.user_ids)
                if len(buf_uids) >= chunk:
                    _flush()
            _flush()
            if n_users == 0:
                # An empty book still writes a valid file with the schema (the
                # writer would otherwise close with no row group at all).
                writer.write_table(pa.table(
                    {"user_id": pa.array([], type=pa.string()),
                     "embedding": pa.array([], type=pa.list_(pa.float32()))}, schema=schema))
        finally:
            writer.close()
            ds.close()
        elapsed = max(time.time() - t0, 1e-6)
        return {"n_users": n_users, "dim": int(self.model.config.dim),
                "elapsed_sec": round(elapsed, 3), "users_per_sec": round(n_users / elapsed, 1),
                "out": str(out_path)}
