"""flash-attn ≡ SDPA numeric equivalence check (GPU only)."""

from __future__ import annotations

from typing import Any


def _check_flash_vs_sdpa() -> dict[str, Any]:
    """Compare flash-attn kernel output to the SDPA fallback on small varlen inputs.

    Only runs when CUDA is available.  Forces the SDPA path by temporarily
    setting ``pragmatiq.models.layers._HAS_FLASH = False``, then restores the
    original value.  Uses ``dropout_p=0.0`` so both paths are deterministic.

    Returns a dict with keys:
        skipped (bool), skip_reason (str),
        flash_available (bool), passed (bool),
        max_abs_diff (float), mean_abs_diff (float), tol (float),
        error (str | None)
    """
    result: dict[str, Any] = {
        "skipped": False,
        "skip_reason": "",
        "flash_available": False,
        "passed": False,
        "max_abs_diff": float("nan"),
        "mean_abs_diff": float("nan"),
        "tol": 1e-2,
        "error": None,
    }

    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            result["skipped"] = True
            result["skip_reason"] = "no CUDA"
            return result

        import pragmatiq.models.layers as _layers  # noqa: PLC0415
        from pragmatiq.models.layers import varlen_self_attention  # noqa: PLC0415

        has_flash_orig: bool = _layers._HAS_FLASH
        result["flash_available"] = has_flash_orig

        if not has_flash_orig:
            result["skipped"] = True
            result["skip_reason"] = "flash unavailable — SDPA only, check skipped"
            return result

        # Build small varlen inputs: 3 segments of lengths [5, 3, 7] → total=15 tokens
        torch.manual_seed(42)
        segment_lens = [5, 3, 7]
        total_t = sum(segment_lens)
        n_heads = 4
        head_dim = 16  # small enough for a quick CPU-equivalent check
        n_seg = len(segment_lens)

        # cu_seqlens: int32 prefix-sum [0, 5, 8, 15]
        cu = torch.zeros(n_seg + 1, dtype=torch.int32, device="cuda")
        for i, ll in enumerate(segment_lens):
            cu[i + 1] = cu[i] + ll
        max_seqlen = max(segment_lens)

        # Random bf16 tensors on CUDA (flash-attn requires bf16/fp16)
        q = torch.randn(total_t, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(total_t, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(total_t, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")

        # (1) Flash-attn path — _HAS_FLASH is True, tensors are CUDA + bf16
        with torch.no_grad():
            out_flash = varlen_self_attention(
                q, k, v, cu, max_seqlen, rope=None, rope_pos=None, dropout_p=0.0
            )
        torch.cuda.synchronize()

        # (2) SDPA fallback — temporarily suppress flash
        _layers._HAS_FLASH = False
        try:
            with torch.no_grad():
                out_sdpa = varlen_self_attention(
                    q, k, v, cu, max_seqlen, rope=None, rope_pos=None, dropout_p=0.0
                )
            torch.cuda.synchronize()
        finally:
            _layers._HAS_FLASH = has_flash_orig  # always restore

        diff = (out_flash.float() - out_sdpa.float()).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        tol = result["tol"]

        result["max_abs_diff"] = max_abs
        result["mean_abs_diff"] = mean_abs
        result["passed"] = max_abs <= tol

        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[flash-check] {status}: max_abs={max_abs:.2e} mean_abs={mean_abs:.2e} "
            f"tol={tol:.0e}",
            flush=True,
        )

    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        result["passed"] = False
        print(f"[flash-check] ERROR: {exc}", flush=True)

    return result
