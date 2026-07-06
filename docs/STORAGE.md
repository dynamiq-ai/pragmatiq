# Storage — URL schemes and object-store support

pragmatiq uses [fsspec](https://filesystem-spec.readthedocs.io/) for all I/O,
so the same API functions that work on local paths also accept remote URLs.
Object-store inputs are staged to a local temp directory before the local pipeline
runs; outputs are uploaded back when the call finishes cleanly.

## Supported URL schemes

| Scheme | Backend | Extra required |
|--------|---------|----------------|
| (no scheme) or `file://` | Local filesystem | — (always available) |
| `memory://` | In-memory fsspec filesystem | — (built-in, useful for tests) |
| `s3://` or `s3a://` | Amazon S3 / S3-compatible | `pip install 'pragmatiq[s3]'` → installs `s3fs` |
| `gs://` or `gcs://` | Google Cloud Storage | `pip install 'pragmatiq[gcs]'` → installs `gcsfs` |
| `az://`, `abfs://`, `abfss://`, `adl://` | Azure Blob Storage | `pip install 'pragmatiq[azure]'` → installs `adlfs` |

Any other scheme supported by an installed fsspec implementation also works.

## Stage-in / stage-out model

The internal pipeline modules (LMDB shards, ONNX runtime, checkpoint files) need
real local filesystem paths and cannot open `s3://` directly.  pragmatiq resolves
this at the `pragmatiq.api` boundary with a **stage-in / stage-out** approach:

1. **Stage-in** — At the start of each API call, remote input URLs
   (`shard_dir`, `run`, `label_path`, `data_dir`, …) are downloaded to a fresh
   per-call temp directory.  Local paths pass through unchanged — zero overhead
   for the common case.

2. **Unchanged local pipeline** — The internal modules (`sharding.py`,
   `dataset.py`, `tokenizer.py`, `pretrainer.py`, model code) run against the
   local staged paths exactly as they would without object-store support.

3. **Stage-out** — On *clean exit* (no exception), registered output directories
   and files are uploaded to their remote destination URLs.  On exception, nothing
   is uploaded and the temp directory is removed — no partial/corrupt artifacts
   land in the store.

The staging context manager lives in `pragmatiq/storage/staging.py` and is
available as `pragmatiq.storage.staging` for advanced use:

```python
from pragmatiq.storage.staging import staging, Stage
```

## Usage examples

```python
import pragmatiq.api as api

# synthesize → remote raw data
api.synthesize({"n_users": 100_000, "seed": 0}, out="s3://my-bucket/raw")

# tokenize from remote input → remote output
api.tokenize("s3://my-bucket/raw", "s3://my-bucket/tok")

# pretrain on remote shards, checkpoints saved remotely
api.pretrain("s3://my-bucket/tok", "run-01", model_size="small",
             runs_root="s3://my-bucket/runs")

# embed with remote run, write parquet to remote storage
api.embed("s3://my-bucket/tok", "s3://my-bucket/runs/run-01",
          out="s3://my-bucket/embeddings.parquet")
```

## Resume from remote

When `resume="auto"` and `runs_root` is a remote URL, pragmatiq checks whether
`runs_root/run_name` already exists in the remote store.  If it does, it is
downloaded into the local staging area *before* training starts, so the trainer
resumes from the existing checkpoint.  On clean exit the updated run directory
(with the new checkpoint) is uploaded back.

## Known limitation

Outputs are uploaded **at the end of the API call**, not streamed live during
training.  This means:

- There is no incremental remote checkpointing during a long pretrain run.  If
  the process is killed mid-run, no checkpoint is written to the remote store.
- For fault-tolerant long runs use a local `runs_root` with a periodic sync
  (e.g. `aws s3 sync`) running alongside, or run on infrastructure with durable
  local disks and sync after the run completes.

This limitation is documented intentionally and will be addressed in a future
release (streaming checkpoint upload after each checkpoint interval).
