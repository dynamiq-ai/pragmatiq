# Software Bill of Materials (SBOM)

This directory contains CycloneDX SBOMs generated per pragmatiq release.

## What is an SBOM?

A Software Bill of Materials lists every package (direct and transitive) that
pragmatiq depends on, together with version numbers, licenses, and checksums.
This is required for supply-chain security reviews, FedRAMP, SOC 2 audits,
and internal BYOC (Bring Your Own Cloud) compliance programmes.

## Format

SBOMs are in [CycloneDX](https://cyclonedx.org/) JSON format (schema v1.4+).
The filename encodes the pragmatiq version:

```
sbom/pragmatiq-<version>.cdx.json
```

## Generation

SBOMs are regenerated automatically in CI for every release by
`scripts/supply_chain/gen_sbom.sh`:

```bash
bash scripts/supply_chain/gen_sbom.sh
```

The script uses [cyclonedx-bom](https://pypi.org/project/cyclonedx-bom/) and
snapshots the active Python environment.

## How to use for compliance review

1. Open `sbom/pragmatiq-<version>.cdx.json` in any CycloneDX-compatible
   viewer (e.g., [OWASP Dependency-Track](https://dependencytrack.org/),
   [Anchore Grype](https://github.com/anchore/grype), or the online viewer
   at <https://sbom.io>).
2. Cross-reference component licenses against your organisation's approved
   license list.
3. Feed the SBOM to a vulnerability scanner (Grype, Trivy, pip-audit) to
   check for known CVEs in the dependency graph.

## Vulnerability scanning

The CI `supply-chain` job runs `pip-audit` on every push:

```bash
pip-audit --desc --strict
```

A failing `pip-audit` blocks the build until the vulnerability is patched or
explicitly exempted with a documented justification.

## License review

```bash
pip-licenses --from=mixed --order=license --format=markdown
```

This lists all packages and their SPDX license identifiers.

## Attribution

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
