"""Isolation tests for the calendar embedding (Phase 4)."""

from __future__ import annotations

import torch

from pragmatiq.models.embeddings import CalendarEmbedding


def _t(x: int) -> torch.Tensor:
    return torch.tensor([x])


def test_calendar_is_periodic_in_each_axis() -> None:
    # sin/cos encodings make the calendar wrap: hour 24≡0, dow 7≡0, dom 32≡1.
    ce = CalendarEmbedding(16).eval()
    base = ce(_t(0), _t(0), _t(1))
    wrapped = ce(_t(24), _t(7), _t(32))
    assert torch.allclose(base, wrapped, atol=1e-5)


def test_calendar_finite_over_full_ranges() -> None:
    ce = CalendarEmbedding(16).eval()
    hour = torch.arange(0, 24)
    dow = torch.arange(0, 7).repeat(4)[:24]
    dom = torch.arange(1, 25)
    out = ce(hour, dow, dom)
    assert out.shape == (24, 16)
    assert torch.isfinite(out).all()


def test_calendar_discriminates_time_of_day() -> None:
    ce = CalendarEmbedding(16).eval()
    night = ce(_t(2), _t(0), _t(1))
    noon = ce(_t(12), _t(0), _t(1))
    assert not torch.allclose(night, noon, atol=1e-3)
