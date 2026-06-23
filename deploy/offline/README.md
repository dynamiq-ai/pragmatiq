# Air-gapped (offline) install

pragmatiq can be installed in environments with no outbound internet access by
pre-downloading wheels on a connected host and transferring the bundle.

## Quick-start

### Step 1 — On a connected host

```bash
bash deploy/offline/build_offline_bundle.sh --extras serve
# or for the full ML stack:
bash deploy/offline/build_offline_bundle.sh --extras both
```

This downloads all required wheels into `./offline_bundle/`.

Alternatively, run pip download directly:

```bash
pip download 'pragmatiq[serve]' -d ./offline_bundle/
# or
pip download 'pragmatiq[full]' -d ./offline_bundle/
```

### Step 2 — Transfer

Copy the `offline_bundle/` directory to the air-gapped host (USB drive, SCP
over a bastion, S3 sync, etc.).

### Step 3 — On the air-gapped host

```bash
pip install --no-index --find-links=./offline_bundle/ 'pragmatiq[serve]'
```

For the full ML stack:

```bash
pip install --no-index --find-links=./offline_bundle/ 'pragmatiq[full]'
```

## Lockfile (uv)

A `uv.lock` file is generated per release in CI:

```bash
uv lock
```

This pins every transitive dependency to an exact version and hash.  The
lockfile is committed to the repository so reproducible installs are possible:

```bash
uv sync --frozen
```

On air-gapped hosts, use the lockfile together with the offline bundle to
ensure byte-for-byte reproducibility.

## extras reference

| Extra   | Contents                                                     |
|---------|--------------------------------------------------------------|
| `serve` | Slim serving stack: torch, numpy, triton model.py only       |
| `train` | Training stack: lightning, torch-geometric, etc.             |
| `full`  | All optional dependencies: train + serve + aml + tracking    |

## Attribution

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
