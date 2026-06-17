> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

## Summary

- 

## Verification

- [ ] `pytest tests/ -x -q`
- [ ] `ruff check . && mypy pragmatiq`
- [ ] Relevant acceptance check: `bash scripts/gates/gate_<N>.sh`

## Notes

- [ ] I updated docs/tests for user-visible behavior changes.
- [ ] I did not add logic to `pragmatiq/cli.py`; CLI changes delegate to `pragmatiq/api.py`.
- [ ] Randomized behavior has explicit seed handling.
- [ ] New docs/assets have clear provenance and do not use Revolut trademarks, Revolut trade dress, or paper figures without attribution.
