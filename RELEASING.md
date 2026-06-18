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

1. Open a **release PR from `develop` into `main`**.
2. In that PR, bump the version following [PEP 440](https://peps.python.org/pep-0440/)
   in all three places (they must agree):
   - `pyproject.toml` (`version = "..."`),
   - `pragmatiq/__init__.py` (the `__version__` fallback),
   - `CITATION.cff` (`version:`).
   Examples: `0.1.0b2` → `0.1.0b3` (next pre-release) or `0.1.0` (first final).
   Update `CHANGELOG.md`.
3. Merge the PR. On the push to `main`, `.github/workflows/release.yml`:
   - reads the version from `pyproject.toml`,
   - if a `v<version>` tag already exists, does nothing — so a merge that does not
     bump the version never publishes,
   - otherwise builds the sdist + wheel, uploads to **PyPI via Trusted
     Publishing**, then creates the `v<version>` tag and a GitHub Release (marked a
     pre-release for `aN`/`bN`/`rcN` versions, otherwise a full release).

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
