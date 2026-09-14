"""RunPod launcher: the pod image and the flash-attn wheel must be built for the
same torch / python, the REST client must identify itself (Cloudflare rejects the
default urllib agent), and the default pipeline delegates to gpu_full_validation."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod_launch.py"


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location("runpod_launch_pairing", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_image_and_flash_wheel_are_paired(rl) -> None:
    torch_major_minor = ".".join(rl.image_torch_version().split(".")[:2])
    m = re.search(r"flash_attn-(?P<v>[\d.]+)\+cu12torch(?P<torch>[\d.]+)cxx11abi(?P<abi>TRUE|FALSE)-"
                  r"(?P<py>cp\d+)-(?P=py)-linux_x86_64\.whl$", rl.FLASH_WHEEL)
    assert m, rl.FLASH_WHEEL
    assert m["torch"] == torch_major_minor
    assert m["py"] == rl.image_python_tag()
    assert m["v"] == rl.FLASH_ATTN_VERSION
    assert m["abi"] == "TRUE"  # torch >= 2.7 CUDA wheels use the C++11 ABI
    assert rl.image_torch_version() == "2.8.0" and rl.image_python_tag() == "cp311"
    assert "cuda12.8" in rl.IMAGE


def test_install_block_keeps_the_image_torch(rl) -> None:
    assert "--no-deps -e ." in rl.INSTALL
    assert 'torch==$TORCH_VER' in rl.INSTALL
    assert rl.FLASH_WHEEL in rl.INSTALL


def test_rest_client_sends_a_user_agent(rl, monkeypatch) -> None:
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"id": "p"}'

    def fake_urlopen(req, timeout=0):
        seen["ua"] = req.get_header("User-agent")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert rl._req("GET", "/pods/p", "k") == {"id": "p"}
    assert seen["ua"].startswith("pragmatiq-runpod-launch/")


def test_default_pipeline_delegates_to_full_validation(rl) -> None:
    cmd = rl.pipeline_command("--tag rc1 --skip-triton", devices_sweep="1,2,4,8", out="outputs/gpu-validation-rc1")
    assert "scripts/gpu_full_validation.py" in cmd
    assert "--devices-sweep 1,2,4,8" in cmd and "--out outputs/gpu-validation-rc1" in cmd
    assert cmd.rstrip().endswith("--tag rc1 --skip-triton")
    assert "api.synthesize" not in cmd  # no second, hand-written pipeline
    assert rl.pipeline_command().rstrip().endswith("gpu_full_validation.py")


def test_pod_body_uses_the_paired_image(rl, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(rl, "_req", lambda m, p, k, body=None: captured.update(body or {}) or {"id": "x"})
    rl.create_pod("k", "NVIDIA A100 80GB PCIe", "n", gpu_count=2)
    assert captured["imageName"] == rl.IMAGE and captured["gpuCount"] == 2
