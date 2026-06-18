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

pragmatiq is currently pre-1.0. Security fixes target the latest commit on the
default branch unless a tagged release states otherwise.

## Data handling

This repository includes a synthetic data generator and examples. Do not attach
real customer data, banking records, credentials, model checkpoints containing
sensitive data, or private aggregate statistics to public issues or pull
requests.
