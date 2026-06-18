"""Muon optimizer behavior beyond a single step."""

from __future__ import annotations

import torch

from pragmatiq.training.optim import Muon


def test_muon_converges_on_a_quadratic() -> None:
    # Minimize ||W - target||^2; the orthogonalized update must make real progress.
    torch.manual_seed(0)
    target = torch.randn(16, 12)
    W = torch.nn.Parameter(torch.zeros(16, 12))
    opt = Muon([W], lr=0.1)

    def loss() -> torch.Tensor:
        return ((W - target) ** 2).sum()

    initial = float(loss().detach())
    for _ in range(100):
        opt.zero_grad()
        loss().backward()
        opt.step()
    assert float(loss().detach()) < 0.5 * initial


def test_muon_zero_grad_is_noop_and_finite() -> None:
    W = torch.nn.Parameter(torch.randn(8, 8))
    before = W.detach().clone()
    opt = Muon([W], lr=0.1)  # default weight_decay=0 → a zero grad must not move W
    W.grad = torch.zeros_like(W)
    opt.step()
    assert torch.isfinite(W).all()
    assert torch.allclose(W, before)


def test_muon_skips_params_without_grad() -> None:
    a = torch.nn.Parameter(torch.randn(4, 4))
    b = torch.nn.Parameter(torch.randn(4, 4))
    opt = Muon([a, b], lr=0.1)
    a.grad = torch.ones_like(a)  # only a has a grad
    a0, b0 = a.detach().clone(), b.detach().clone()
    opt.step()
    assert torch.allclose(b, b0)  # b untouched (p.grad is None → continue)
    assert not torch.allclose(a, a0) and torch.isfinite(a).all()  # a stepped, stays finite
