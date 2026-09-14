# Releasing pragmatiq

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

pragmatiq releases from `main`:

- **`main`** — the release branch and the base for pull requests. Feature,
  fix and release branches (`feat/...`, `fix/...`, `release/x.y.z`) are opened
  against `main` and squash-merged as one commit (a release PR's commit is
  titled `pragmatiq x.y.z (#N)`). Every version-bumping merge to `main`
  publishes a release to PyPI and GitHub.
- **`develop`** — mirrors `main`; it is fast-forwarded after each merge and
  exists for tooling that expects a `develop` branch.

```
feature / release branch ──PR (squash)──▶ main ──▶ tag + GitHub Release ──▶ PyPI ──▶ fast-forward develop
```

## Cutting a release

### 1. Bump the version

In the release PR, update the version following [PEP 440](https://peps.python.org/pep-0440/)
in **all three places** (they must agree):

- `pyproject.toml` — `version = "..."`
- `pragmatiq/__init__.py` — the `__version__` fallback string
- `CITATION.cff` — `version:` field

Add a `## [X.Y.Z]` entry to `CHANGELOG.md` (newest at top) with *Breaking*
(each break with a migration line — the pre-2.0 policy in `docs/STABILITY.md`),
*Added*, *Changed*, *Removed*; update the "Breaks in x.y.z" table in
`docs/STABILITY.md` and the contract goldens in `tests/contract/` for any
signature or default that changed. The version bump is the **last** change in
the release PR, because `release.yml` publishes on any push to `main` whose
version has no tag yet.

### 2. Regenerate the lock file and the docs facts

```bash
uv lock
python scripts/docs_facts.py        # website/data/facts.json (drift-checked in CI)
python scripts/docs_drift_check.py
```

Commit the updated `uv.lock` and `facts.json` alongside the version bump.

### 3. Regenerate the SBOM

```bash
bash scripts/supply_chain/gen_sbom.sh
```

The SBOM lands in `dist/sbom/` (not committed); attach it to the GitHub Release
assets. The CI supply-chain job regenerates it on every push to prove the
generator works.

### 4. Run the full validation suite

```bash
bash scripts/gates/run_full_validation.sh
```

All gates must be green before merging: `gate_1` … `gate_8`,
`gate_9_contract`, `gate_serve_slim`, `gate_storage`, `gate_integrations`,
`gate_10_byoc` (CI runs every one). Run the GPU validation on a rented pod
(`python scripts/runpod_launch.py ...`, legs in `scripts/gpuval/`) and commit
its JSON under `docs/benchmarks/` so the README `GPU_VALIDATION_RESULTS` block
reflects the release; `bash scripts/deploy_serving.sh` on the pod must print
`SERVING SMOKE GREEN`.

### 5. Merge and tag

Merge the release PR into `main`. On the push to `main`, `.github/workflows/release.yml`:

- reads the version from `pyproject.toml`,
- if a `v<version>` tag already exists, does nothing — so a merge that does not
  bump the version never publishes,
- otherwise builds the sdist + wheel, uploads to **PyPI via Trusted Publishing**,
  then creates the `v<version>` tag and a GitHub Release (marked a pre-release for
  `aN`/`bN`/`rcN` versions, otherwise a full release).

It is one self-contained job by design: a Release created by the built-in
`GITHUB_TOKEN` does not trigger other workflows, so publishing and tagging
cannot be split across chained workflows.

## One-time PyPI setup

Trusted Publishing needs no API token or secret. On
<https://pypi.org/manage/project/pragmatiq/settings/publishing/>, add a trusted
publisher with:

| field | value |
| --- | --- |
| owner | `dynamiq-ai` |
| repository | `pragmatiq` |
| workflow | `release.yml` |
| environment | `pypi` |

## Hotfixes

For an urgent fix to a published release, branch from `main`, open a PR back into
`main` with a patch-version bump, merge it (which releases), then fast-forward
`develop` so the branches stay in sync.

## What counts as BREAKING vs MINOR vs PATCH

See [`docs/STABILITY.md`](docs/STABILITY.md) for the pre-2.0 policy. Quick
reference:

| Change | Bump |
| --- | --- |
| Serving input/output name or dtype; checkpoint format version | MAJOR |
| Rename / remove an `api.*` function, CLI command or param; change a default; shard/generator format | MINOR, listed under *Breaking* with a migration line |
| New `api.*` function, new optional param, new return key, new extra | MINOR |
| Change a `# GUESS` default value (for new runs only) | MINOR |
| Bugfix / internal refactor / perf improvement | PATCH |
