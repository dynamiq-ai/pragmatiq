# Releasing pragmatiq

pragmatiq uses a two-branch flow:

- **`develop`** — the integration branch and the default base for pull requests.
  Feature and fix branches (`feat/...`, `fix/...`) are opened against `develop`.
- **`main`** — the release branch. Every version-bumping merge to `main` publishes
  a release to PyPI and GitHub.

```
feature branch ──PR──▶ develop ──release PR──▶ main ──▶ tag + GitHub Release ──▶ PyPI
```

## Cutting a release

### 1. Bump the version

In the release PR, update the version following [PEP 440](https://peps.python.org/pep-0440/)
in **all three places** (they must agree):

- `pyproject.toml` — `version = "..."`
- `pragmatiq/__init__.py` — the `__version__` fallback string
- `CITATION.cff` — `version:` field

Add a `## [X.Y.Z]` entry to `CHANGELOG.md` (newest at top).

### 2. Regenerate the lock file

```bash
uv lock
```

Commit the updated `uv.lock` alongside the version bump.

### 3. Regenerate the SBOM

```bash
bash scripts/supply_chain/gen_sbom.sh
```

Commit the updated `sbom/` output. The CI supply-chain job verifies the SBOM on
every push; a stale SBOM will fail CI.

### 4. Run the full validation suite

```bash
bash scripts/gates/run_full_validation.sh
```

All gates must be green before merging. Docker-dependent gates (gate_7 Triton)
require a Docker daemon; run them locally or in CI with Docker enabled. The
offline-capable gates (gate_1 through gate_6, gate_8, gate_9_contract,
gate_storage, gate_integrations, gate_10_byoc) must all pass without Docker.

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
`main` with a patch-version bump, merge it (which releases), then merge `main`
back into `develop` so the branches stay in sync.

## What counts as BREAKING vs MINOR vs PATCH

See [`docs/STABILITY.md`](docs/STABILITY.md) for the full SemVer policy. Quick
reference:

| Change | Bump |
| --- | --- |
| Rename / remove an `api.*` function, CLI command, or serving input/output name | MAJOR |
| Checkpoint format version increment | MAJOR |
| New `api.*` function, new optional param, new return key, new extra | MINOR |
| Change a `# GUESS` default value (for new runs only) | MINOR |
| Bugfix / internal refactor / perf improvement | PATCH |
