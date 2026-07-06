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

**The bundle is platform-specific.** By default, wheels are resolved for the
*build* host's OS/architecture and Python version (the script prints this
assumption loudly), so the bundle only installs on an air-gapped host that
matches. If the air-gapped host differs — e.g. you build on macOS or Python
3.12 but deploy to Linux/Python 3.11 — pass the target explicitly:

```bash
bash deploy/offline/build_offline_bundle.sh --extras serve \
  --platform manylinux2014_x86_64 --python-version 3.11 --abi cp311
```

`--platform` is repeatable for additional compatible tags (e.g. also
`--platform manylinux_2_17_x86_64`). Cross-platform downloads pass pip's
`--only-binary=:all:` (pip requires it with `--platform`/`--python-version`/
`--abi`), so every dependency must publish a wheel for the target — an
sdist-only dependency fails the download rather than producing a broken bundle.

Alternatively, run pip download directly:

```bash
pip download 'pragmatiq[serve]' -d ./offline_bundle/
# or
pip download 'pragmatiq[full]' -d ./offline_bundle/
# cross-platform equivalent:
pip download 'pragmatiq[serve]' -d ./offline_bundle/ --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 3.11 --abi cp311
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

| Extra   | Contents                                                              |
|---------|-----------------------------------------------------------------------|
| `serve` | Slim serving stack: ONNX / ONNX Runtime / Triton client (no Lightning / torch-geometric / transformers) |
| `train` | Training stack: Lightning + matplotlib                                |
| `full`  | All optional dependencies: data, train, serve, aml, text, tracking, gbdt, demo, s3, gcs, azure |

## Attribution

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
