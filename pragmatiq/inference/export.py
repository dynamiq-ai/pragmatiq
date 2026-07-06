"""Model export.

``export_onnx`` exports a **dense reformulation** of the trained pragmatiq model:
the same ``nn.Module`` weights run over static-rectangular padded tensors instead
of the native varlen (no-padding) layout, so the graph is a clean ``torch.export``
target. The dense forward reuses ``model.embed`` / ``model.calendar`` and the
event / profile / history encoders, masking padded positions on the key axis so
per-user outputs are numerically equal to :meth:`PragmaModel.embed_users`.

The Triton **python backend** (``deploy/triton``) remains the high-throughput
serving path because it runs the native varlen model and skips the padding the
dense graph materializes — a deployment choice, not a fidelity gap.

Scope: the dense graph restates the BPE token path. PRAGMA+Nemotron models
(``text_value_mode="embed"``) additionally route text-field values through a
frozen text encoder; that contribution is not part of the dense reformulation, so
:func:`export_onnx` rejects an embed-mode model and points it at the native Triton
backend rather than emitting a graph whose embeddings would diverge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from ..data.collate import PackedBatch
from ..data.tokenizer import EVT, USR
from ..models.layers import Encoder, TransformerBlock, VarlenAttention
from ..models.pragmatiq import PragmaModel

# Order of the dense tensors fed to :class:`DenseEmbedder` and named in the ONNX
# graph; ``export_onnx`` and any runtime feed dict use the same order.
DENSE_INPUT_NAMES = [
    "event_key_ids", "event_value_ids", "event_positions", "event_token_mask",
    "event_mask", "event_time_log", "event_hour", "event_dow", "event_dom",
    "profile_key_ids", "profile_value_ids", "profile_positions",
    "profile_token_mask", "profile_time_log",
]


def pack_to_dense(batch: PackedBatch) -> dict[str, torch.Tensor]:
    """Convert a varlen :class:`PackedBatch` to static-rectangular dense tensors.

    Returns a dict keyed by :data:`DENSE_INPUT_NAMES`:

    - events ``[U, Emax, Lmax]`` for key / value / position ids (long) plus a
      float token-mask ``[U, Emax, Lmax]``, an event-mask ``[U, Emax]`` and the
      per-event calendar / time features ``[U, Emax]`` (time-log, hour, dow, dom);
    - profile ``[U, Ptot]`` for key / value / position ids (long), a float
      token-mask ``[U, Ptot]`` and the per-token log-seconds ``[U, Ptot]``.

    Padding is zeros; validity is carried entirely in the float masks so the
    padded rows never influence any user's embedding.
    """
    U = batch.n_users
    cu_e = batch.cu_seqlens_event
    ev_per_user = (batch.cu_seqlens_history[1:] - batch.cu_seqlens_history[:-1]).long()
    ev_lens = (cu_e[1:] - cu_e[:-1]).long()
    emax = max(int(ev_per_user.max()) if U else 1, 1)
    lmax = max(int(ev_lens.max()) if ev_lens.numel() else 1, 1)

    event_key_ids = torch.zeros(U, emax, lmax, dtype=torch.long)
    event_value_ids = torch.zeros(U, emax, lmax, dtype=torch.long)
    event_positions = torch.zeros(U, emax, lmax, dtype=torch.long)
    event_token_mask = torch.zeros(U, emax, lmax, dtype=torch.float32)
    event_mask = torch.zeros(U, emax, dtype=torch.float32)
    event_time_log = torch.zeros(U, emax, dtype=torch.float32)
    event_hour = torch.zeros(U, emax, dtype=torch.long)
    event_dow = torch.zeros(U, emax, dtype=torch.long)
    event_dom = torch.zeros(U, emax, dtype=torch.long)

    g_event = 0
    for u in range(U):
        for e in range(int(ev_per_user[u])):
            lo, hi = int(cu_e[g_event]), int(cu_e[g_event + 1])
            n = hi - lo
            event_key_ids[u, e, :n] = batch.key_ids[lo:hi]
            event_value_ids[u, e, :n] = batch.value_ids[lo:hi]
            event_positions[u, e, :n] = batch.positions[lo:hi]
            event_token_mask[u, e, :n] = 1.0
            event_mask[u, e] = 1.0
            event_time_log[u, e] = batch.event_time_log[g_event]
            event_hour[u, e] = batch.event_hour[g_event]
            event_dow[u, e] = batch.event_dow[g_event]
            event_dom[u, e] = batch.event_dom[g_event]
            g_event += 1

    cu_pi = batch.cu_seqlens_profile_item
    items_per_user = (batch.cu_seqlens_profile[1:] - batch.cu_seqlens_profile[:-1]).long()
    pi_lens = (cu_pi[1:] - cu_pi[:-1]).long()
    tok_per_user = torch.zeros(U, dtype=torch.long)
    g_item = 0
    for u in range(U):
        for _ in range(int(items_per_user[u])):
            tok_per_user[u] += pi_lens[g_item]
            g_item += 1
    ptot = max(int(tok_per_user.max()) if U else 1, 1)

    profile_key_ids = torch.zeros(U, ptot, dtype=torch.long)
    profile_value_ids = torch.zeros(U, ptot, dtype=torch.long)
    profile_positions = torch.zeros(U, ptot, dtype=torch.long)
    profile_token_mask = torch.zeros(U, ptot, dtype=torch.float32)
    profile_time_log = torch.zeros(U, ptot, dtype=torch.float32)

    g_item = 0
    for u in range(U):
        cursor = 0
        for _ in range(int(items_per_user[u])):
            lo, hi = int(cu_pi[g_item]), int(cu_pi[g_item + 1])
            n = hi - lo
            profile_key_ids[u, cursor:cursor + n] = batch.prof_key_ids[lo:hi]
            profile_value_ids[u, cursor:cursor + n] = batch.prof_value_ids[lo:hi]
            profile_positions[u, cursor:cursor + n] = batch.prof_positions[lo:hi]
            profile_time_log[u, cursor:cursor + n] = batch.prof_time_log[g_item]
            profile_token_mask[u, cursor:cursor + n] = 1.0
            cursor += n
            g_item += 1

    return {
        "event_key_ids": event_key_ids, "event_value_ids": event_value_ids,
        "event_positions": event_positions, "event_token_mask": event_token_mask,
        "event_mask": event_mask, "event_time_log": event_time_log,
        "event_hour": event_hour, "event_dow": event_dow, "event_dom": event_dom,
        "profile_key_ids": profile_key_ids, "profile_value_ids": profile_value_ids,
        "profile_positions": profile_positions, "profile_token_mask": profile_token_mask,
        "profile_time_log": profile_time_log,
    }


def _rope_angles(rope: Any, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``TimeRoPE.angles`` for per-row positions ``[B, L]`` → cos/sin ``[B, L, hd]``."""
    freqs = position[..., None].float() * rope.inv_freq[None, None, :]  # [B, L, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply TimeRoPE to ``x`` ``[B, L, H, hd]`` given cos/sin ``[B, L, hd]``."""
    cos = cos[:, :, None, :].to(x.dtype)
    sin = sin[:, :, None, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def _dense_attention(
    attn: VarlenAttention,
    x: torch.Tensor,
    key_bias: torch.Tensor,
    rope_pos: torch.Tensor | None,
) -> torch.Tensor:
    """Dense mirror of :class:`VarlenAttention` over ``[B, L, d]``.

    ``key_bias`` ``[B, 1, 1, L]`` is the additive float mask that zeroes
    attention to padded keys; ``rope_pos`` ``[B, L]`` carries TimeRoPE positions.
    """
    b, length, d = x.shape
    qkv = attn.qkv(x).view(b, length, 3, attn.n_heads, attn.head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    if attn.rope is not None and rope_pos is not None:
        cos, sin = _rope_angles(attn.rope, rope_pos)
        q = _rope_rotate(q, cos, sin)
        k = _rope_rotate(k, cos, sin)
    q = q.permute(0, 2, 1, 3)  # [B, H, L, hd]
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_bias)
    out = out.permute(0, 2, 1, 3).reshape(b, length, d)
    return attn.out(out)


def _dense_block(
    block: TransformerBlock,
    x: torch.Tensor,
    key_bias: torch.Tensor,
    rope_pos: torch.Tensor | None,
) -> torch.Tensor:
    """Dense mirror of :class:`TransformerBlock` (pre-norm attn + GELU MLP)."""
    x = x + _dense_attention(block.attn, block.norm1(x), key_bias, rope_pos)
    x = x + block.ffn(block.norm2(x))
    return x


def _run_encoder(
    encoder: Encoder,
    x: torch.Tensor,
    valid: torch.Tensor,
    rope_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense mirror of :class:`Encoder` over ``[B, L, d]``.

    ``valid`` ``[B, L]`` is 1 on real tokens, 0 on padding; it becomes an
    additive float key bias (``-inf`` on padding). A float bias is used rather
    than a boolean mask because ONNX runtimes reject boolean SDPA masks.
    """
    key_bias = ((1.0 - valid) * torch.finfo(torch.float32).min)[:, None, None, :]
    for block in encoder.blocks:
        x = _dense_block(cast(TransformerBlock, block), x, key_bias, rope_pos)
    return encoder.norm(x)


class DenseEmbedder(nn.Module):
    """Dense (padded) reformulation of :class:`PragmaModel` for ONNX export.

    Consumes the tensors from :func:`pack_to_dense` and returns user embeddings
    ``[U, d]`` numerically equal to :meth:`PragmaModel.embed_users`. The encoders,
    embedding table and calendar MLP are the model's own modules; only the
    block-diagonal varlen attention is restated as masked dense attention.
    """

    def __init__(self, model: PragmaModel) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        event_key_ids: torch.Tensor,  # [U, Emax, Lmax]
        event_value_ids: torch.Tensor,  # [U, Emax, Lmax]
        event_positions: torch.Tensor,  # [U, Emax, Lmax]
        event_token_mask: torch.Tensor,  # [U, Emax, Lmax]
        event_mask: torch.Tensor,  # [U, Emax]
        event_time_log: torch.Tensor,  # [U, Emax]
        event_hour: torch.Tensor,  # [U, Emax]
        event_dow: torch.Tensor,  # [U, Emax]
        event_dom: torch.Tensor,  # [U, Emax]
        profile_key_ids: torch.Tensor,  # [U, Ptot]
        profile_value_ids: torch.Tensor,  # [U, Ptot]
        profile_positions: torch.Tensor,  # [U, Ptot]
        profile_token_mask: torch.Tensor,  # [U, Ptot]
        profile_time_log: torch.Tensor,  # [U, Ptot]
    ) -> torch.Tensor:
        """Return user embeddings ``[U, d]`` (``z_h`` at each user's ``[USR]``)."""
        m = self.model
        d = m.config.dim
        u, emax, lmax = event_key_ids.shape

        # Event encoder: fold [U, Emax] into N rows, prepend a per-event [EVT].
        n = u * emax
        x_tok = m.embed(
            event_key_ids.reshape(n, lmax),
            event_value_ids.reshape(n, lmax),
            event_positions.reshape(n, lmax),
        )  # [N, Lmax, d]
        evt_prefix = m.embed.embed(torch.full((n, 1), EVT, dtype=torch.long))
        x = torch.cat([evt_prefix, x_tok], dim=1)  # [N, 1 + Lmax, d]
        token_valid = event_token_mask.reshape(n, lmax)
        valid = torch.cat([torch.ones(n, 1), token_valid], dim=1)
        h = _run_encoder(m.event_encoder, x, valid)
        evt_vec = h[:, 0].reshape(u, emax, d)  # per-event [EVT] output
        z_e = evt_vec + m.calendar(event_hour, event_dow, event_dom)  # [U, Emax, d]

        # Profile encoder: prepend [USR] anchored at log-second 0.
        x_prof = m.embed(profile_key_ids, profile_value_ids, profile_positions)
        usr_prefix = m.embed.embed(torch.full((u, 1), USR, dtype=torch.long))
        x = torch.cat([usr_prefix, x_prof], dim=1)  # [U, 1 + Ptot, d]
        valid = torch.cat([torch.ones(u, 1), profile_token_mask], dim=1)
        rope_pos = torch.cat([torch.zeros(u, 1), profile_time_log], dim=1)
        z_a = _run_encoder(m.profile_encoder, x, valid, rope_pos)[:, 0]  # [U, d]

        # History encoder: sequence [z_a (USR), z_e…] with the varlen RoPE times.
        x = torch.cat([z_a[:, None, :], z_e], dim=1)  # [U, 1 + Emax, d]
        valid = torch.cat([torch.ones(u, 1), event_mask], dim=1)
        rope_pos = torch.cat([torch.zeros(u, 1), event_time_log], dim=1)
        z_h = _run_encoder(m.history_encoder, x, valid, rope_pos)
        return z_h[:, 0]  # user embedding at the [USR] slot [U, d]


# Tensors sharing each dynamic axis: (axis, group of input names) — the users
# axis spans every tensor; the event / token / profile axes group by structure.
_DYNAMIC_AXES: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, tuple(DENSE_INPUT_NAMES)),  # users (U), present on all inputs
    (1, ("event_key_ids", "event_value_ids", "event_positions", "event_token_mask",
          "event_mask", "event_time_log", "event_hour", "event_dow", "event_dom")),  # events (Emax)
    (2, ("event_key_ids", "event_value_ids", "event_positions", "event_token_mask")),  # event tokens (Lmax)
    (1, ("profile_key_ids", "profile_value_ids", "profile_positions",
          "profile_token_mask", "profile_time_log")),  # profile tokens (Ptot)
)


def _expand_example(dense: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Grow each dynamic axis to at least two so it traces as truly dynamic.

    Padding (zeros, with zeroed masks) never affects outputs, but a dynamic axis
    whose example length is 1 can be folded to a constant during ONNX shape
    inference; padding the tracing example to length two keeps every axis free.
    """
    out = dict(dense)
    for axis, names in _DYNAMIC_AXES:
        if out[names[0]].shape[axis] >= 2:
            continue
        for name in names:
            t = out[name]
            out[name] = torch.cat([t, torch.zeros_like(t.narrow(axis, 0, 1))], dim=axis)
    return out


def export_onnx(model: PragmaModel, example_batch: Any, out_path: str | Path,
                opset: int = 18) -> dict[str, Any]:
    """Export the dense embedder to ONNX from one example :class:`PackedBatch`.

    The graph is shape-dynamic in the user / event / token / profile axes, so a
    single export serves any batch. The round-trip is validated against
    onnxruntime before returning. Returns a manifest ``{out, opset}``. Requires
    the ``serve`` extra (``pip install pragmatiq[serve]``) for ``onnxscript``.

    Supports the BPE token path; embed-mode (PRAGMA+Nemotron) models are rejected
    with a clear error (serve them via the native Triton backend).
    """
    if getattr(model, "text_encoder", None) is not None:
        raise NotImplementedError(
            "ONNX dense export covers the BPE token path. This model carries a frozen "
            "text encoder (PRAGMA+Nemotron, text_value_mode='embed'), whose contribution "
            "to event tokens the dense graph does not restate — serve embed-mode models "
            "with the native Triton python backend (deploy/triton)."
        )
    try:
        import onnxscript  # noqa: F401  (torch dynamo ONNX exporter dependency)
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        from pragmatiq.core.errors import MissingExtraError
        raise MissingExtraError.for_extra("serve", "onnxscript") from e

    dense = pack_to_dense(example_batch)
    args = tuple(_expand_example(dense)[name] for name in DENSE_INPUT_NAMES)

    dim_u = torch.export.Dim("users", min=1)
    dim_e = torch.export.Dim("events", min=1)
    dim_l = torch.export.Dim("event_tokens", min=1)
    dim_p = torch.export.Dim("profile_tokens", min=1)
    event3d = {0: dim_u, 1: dim_e, 2: dim_l}
    event2d = {0: dim_u, 1: dim_e}
    profile2d = {0: dim_u, 1: dim_p}
    dynamic_shapes = {
        "event_key_ids": event3d, "event_value_ids": event3d, "event_positions": event3d,
        "event_token_mask": event3d, "event_mask": event2d, "event_time_log": event2d,
        "event_hour": event2d, "event_dow": event2d, "event_dom": event2d,
        "profile_key_ids": profile2d, "profile_value_ids": profile2d,
        "profile_positions": profile2d, "profile_token_mask": profile2d,
        "profile_time_log": profile2d,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        DenseEmbedder(model), args, str(out_path),
        input_names=DENSE_INPUT_NAMES, output_names=["user_embedding"],
        opset_version=opset, dynamo=True, dynamic_shapes=dynamic_shapes,
    )

    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        from pragmatiq.core.errors import MissingExtraError
        raise MissingExtraError.for_extra("serve", "onnxruntime") from e

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feeds = {name: dense[name].numpy() for name in DENSE_INPUT_NAMES}
    onnx_out = session.run(None, feeds)[0]
    # Validate against the varlen path itself: the dense reformulation must be
    # numerically equivalent to the varlen forward, on the same example batch.
    with torch.no_grad():
        native = model.embed_users(example_batch).detach().cpu().numpy()
    max_abs_diff = float(np.abs(onnx_out - native).max())
    if not np.allclose(onnx_out, native, atol=1e-3):
        raise RuntimeError(
            f"exported ONNX is not numerically equivalent to the varlen path "
            f"(max abs diff {max_abs_diff:.2e} > 1e-3) — refusing to ship {out_path}"
        )
    return {"out": str(out_path), "opset": opset, "max_abs_diff": max_abs_diff}
