UI seam. Apps consume pragmatiq's public API (`pragmatiq.api`) or the serving
HTTP API only; they MUST NOT import pragmatiq.training.

## Contents

- `apps/demo/` — Streamlit demo (`app.py`). Pick a synthetic user → event
  timeline → embedding computed live via `PragmaModel.embed_records` → ego
  transfer graph. Requires the `[demo]` extra (`pip install -e ".[demo]"`).
  Run with `streamlit run apps/demo/app.py` after `pragmatiq quickstart`.
