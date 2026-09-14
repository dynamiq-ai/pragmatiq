# Security Policy

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

## Reporting a vulnerability

Please report suspected vulnerabilities privately by emailing
`support@getdynamiq.ai`. Include:

- affected version or commit,
- steps to reproduce,
- expected impact,
- whether the issue affects generated synthetic data, model training, serving,
  or repository infrastructure.

Please do not open a public GitHub issue for a vulnerability before maintainers
have had a chance to triage it.

## Supported versions

Security fixes target the latest release on PyPI and `main`; older releases
are not patched.

## Data handling

This repository includes a synthetic data generator and examples. Do not attach
real customer data, banking records, credentials, model checkpoints containing
sensitive data, or private aggregate statistics to public issues or pull
requests.

## Supply chain

- **No phone-home.** The library never contacts the network unless you pass a
  remote URL (object storage) or install and configure a tracking extra;
  `scripts/supply_chain/no_phone_home.py` and
  `tests/boundaries/test_no_phone_home.py` prove it (gate 10).
- **SBOM.** `bash scripts/supply_chain/gen_sbom.sh` writes a CycloneDX JSON
  SBOM of the active environment to `dist/sbom/pragmatiq-<version>.cdx.json`.
  It is generated in CI on every push (`supply-chain` job) and attached to each
  GitHub Release; feed it to Dependency-Track, Grype, or Trivy for your own
  review.
- **Serving surface.** The Triton backend validates every request record and
  caps a request at `PRAGMATIQ_SERVE_MAX_RECORDS` users (default 1024), so one
  payload cannot exhaust the device; checkpoints load with
  `torch.load(weights_only=True)` and only fall back to a full unpickle, with a
  warning, for files you trust. Set `PRAGMATIQ_SERVE_CPU=1` to keep a shared
  GPU host's devices away from the serving container.
- **Vulnerability and license scans.** The CI `supply-chain` job runs
  `pip-audit --strict` and `pip-licenses` on the resolved dependency set; a
  failing audit blocks the build until the dependency is patched or the
  advisory is exempted with a documented reason in `.github/workflows/ci.yml`.
