"""Failure-safety plumbing of the RunPod launcher.

Regression guards for the review findings F18-F21/F24: `_req` must raise a
catchable exception (never SystemExit, which escapes `except Exception` and
silently kills the watchdog daemon thread), SSH polling must tolerate
transient API failures, the artifact pull must exclude bulk paths and never
leave a truncated pulled.tar.gz, and the watchdog must kill SSH first and
only hard-DELETE the pod after the cleanup grace window.

The launcher lives in scripts/ (not the package), so it is loaded by path;
its module-level imports are stdlib-only.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import subprocess
import tarfile
import threading
import time
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod_launch.py"


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location("runpod_launch_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# F18: _req error propagation — RunPodAPIError, never SystemExit
# ---------------------------------------------------------------------------
class TestReqErrors:
    def test_http_error_raises_api_error(self, rl, monkeypatch) -> None:
        def boom(req, timeout):  # noqa: ARG001
            raise urllib.error.HTTPError(
                "https://rest.runpod.io/v1/pods", 402, "Payment Required",
                {}, io.BytesIO(b"insufficient balance"),
            )

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(rl.RunPodAPIError, match="402.*insufficient balance"):
            rl._req("POST", "/pods", "k", {"name": "x"})

    def test_url_error_raises_api_error(self, rl, monkeypatch) -> None:
        def boom(req, timeout):  # noqa: ARG001
            raise urllib.error.URLError("egress blocked")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(rl.RunPodAPIError, match="cannot reach"):
            rl._req("GET", "/pods/abc", "k")

    def test_api_error_is_plain_exception_not_systemexit(self, rl) -> None:
        # SystemExit inherits from BaseException only; the watchdog and
        # _terminate_pod rely on `except Exception` catching API failures.
        assert issubclass(rl.RunPodAPIError, Exception)
        assert not issubclass(rl.RunPodAPIError, SystemExit)

    def test_terminate_pod_swallows_api_error(self, rl, monkeypatch, capsys) -> None:
        def failing_req(method, path, key, body=None):  # noqa: ARG001
            raise rl.RunPodAPIError("RunPod API 500: boom")

        monkeypatch.setattr(rl, "_req", failing_req)
        rl._terminate_pod("pod123", "k")  # must not raise
        err = capsys.readouterr().err
        assert "POD TERMINATION FAILED" in err
        assert "pod123" in err


# ---------------------------------------------------------------------------
# F19: SSH polling tolerates transient API failures
# ---------------------------------------------------------------------------
class TestWaitForSsh:
    def test_transient_failures_then_ready(self, rl) -> None:
        calls = {"n": 0}
        sleeps: list[float] = []

        def req(method, path, key):  # noqa: ARG001
            calls["n"] += 1
            if calls["n"] <= 2:
                raise rl.RunPodAPIError("RunPod API 429: rate limited")
            return {"publicIp": "1.2.3.4", "portMappings": {"22": 40022}}

        got = rl._wait_for_ssh("p", "k", attempts=10, interval=1.0,
                               req=req, sleep=sleeps.append)
        assert got == ("1.2.3.4", 40022)
        assert calls["n"] == 3
        assert len(sleeps) == 2  # backed-off retries, not aborts

    def test_never_ready_returns_none(self, rl) -> None:
        got = rl._wait_for_ssh("p", "k", attempts=3, interval=0.0,
                               req=lambda *a: {}, sleep=lambda s: None)
        assert got is None

    def test_all_failures_returns_none_without_raising(self, rl) -> None:
        def req(method, path, key):  # noqa: ARG001
            raise rl.RunPodAPIError("down")

        got = rl._wait_for_ssh("p", "k", attempts=3, interval=0.0,
                               req=req, sleep=lambda s: None)
        assert got is None


# ---------------------------------------------------------------------------
# F21: pull command construction and truncation safety
# ---------------------------------------------------------------------------
class TestBuildPullCmd:
    def test_default_excludes_present_before_glob(self, rl) -> None:
        cmd = rl._build_pull_cmd("outputs/gpu-validation-*")
        assert cmd.startswith("cd /workspace/pragmatiq && tar czf - ")
        assert "--exclude='*/checkpoints'" in cmd
        assert "--exclude='*/data'" in cmd
        # GNU tar applies --exclude only to operands that follow it.
        assert cmd.index("--exclude") < cmd.index("outputs/gpu-validation-*")

    def test_custom_excludes(self, rl) -> None:
        cmd = rl._build_pull_cmd("REPORT.md", excludes=("*.tmp",))
        assert "--exclude='*.tmp'" in cmd
        assert "checkpoints" not in cmd


def _tar_gz_bytes(name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class TestPullArtifacts:
    SSH_BASE = ["ssh", "-p", "22", "root@1.2.3.4"]

    def test_success_renames_partial_and_extracts(self, rl, monkeypatch, tmp_path) -> None:
        real_run = subprocess.run
        seen_cmds: list[list[str]] = []

        def fake_run(cmd, **kw):
            seen_cmds.append(cmd)
            if cmd[0] == "tar":  # local extraction — run for real
                return real_run(cmd, **kw)
            kw["stdout"].write(_tar_gz_bytes("REPORT.md", b"verdict: pass\n"))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(rl.subprocess, "run", fake_run)
        ok = rl._pull_artifacts(self.SSH_BASE, "outputs/*", str(tmp_path),
                                timeout_sec=5.0)
        assert ok
        assert (tmp_path / "pulled.tar.gz").exists()
        assert not (tmp_path / "pulled.tar.gz.part").exists()
        assert (tmp_path / "REPORT.md").read_bytes() == b"verdict: pass\n"
        # The remote command carries the default bulk excludes.
        assert "--exclude='*/checkpoints'" in seen_cmds[0][-1]

    def test_timeout_leaves_no_corrupt_archive(self, rl, monkeypatch, tmp_path) -> None:
        def fake_run(cmd, **kw):
            # Simulate a stream cut off mid-transfer: some bytes then timeout.
            kw["stdout"].write(gzip.compress(b"x" * 4096)[:100])
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

        monkeypatch.setattr(rl.subprocess, "run", fake_run)
        ok = rl._pull_artifacts(self.SSH_BASE, "outputs/*", str(tmp_path),
                                timeout_sec=0.1)
        assert not ok
        assert not (tmp_path / "pulled.tar.gz").exists()
        assert not (tmp_path / "pulled.tar.gz.part").exists()

    def test_empty_glob_match_is_warning_not_archive(self, rl, monkeypatch,
                                                     tmp_path, capsys) -> None:
        def fake_run(cmd, **kw):  # noqa: ARG001
            return subprocess.CompletedProcess(cmd, 0)  # writes nothing

        monkeypatch.setattr(rl.subprocess, "run", fake_run)
        ok = rl._pull_artifacts(self.SSH_BASE, "nope/*", str(tmp_path),
                                timeout_sec=5.0)
        assert not ok
        assert "no files matched" in capsys.readouterr().err
        assert not (tmp_path / "pulled.tar.gz").exists()
        assert not (tmp_path / "pulled.tar.gz.part").exists()

    def test_custom_timeout_is_forwarded(self, rl, monkeypatch, tmp_path) -> None:
        timeouts: list[float | None] = []

        def fake_run(cmd, **kw):
            timeouts.append(kw.get("timeout"))
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

        monkeypatch.setattr(rl.subprocess, "run", fake_run)
        rl._pull_artifacts(self.SSH_BASE, "outputs/*", str(tmp_path),
                           timeout_sec=7200.0)
        assert timeouts == [7200.0]


# ---------------------------------------------------------------------------
# F18/F20: watchdog ordering — SSH kill first, pod DELETE only after grace
# ---------------------------------------------------------------------------
class _FakeProc:
    def __init__(self, order: list[str], fail_kill: bool = False) -> None:
        self._order = order
        self._fail_kill = fail_kill

    def kill(self) -> None:
        if self._fail_kill:
            raise OSError("kill failed")
        self._order.append("kill")


class TestWatchdog:
    def _run(self, rl, *, proc, cleanup_preset: bool, grace_sec: float,
             done_preset: bool = False):
        order: list[str] = []
        if proc is not None:
            proc = proc(order)
        done = threading.Event()
        fired = threading.Event()
        cleanup = threading.Event()
        if done_preset:
            done.set()
        if cleanup_preset:
            cleanup.set()

        def terminate(pod_id: str, key: str) -> None:  # noqa: ARG001
            order.append("terminate")

        t = rl._start_pod_watchdog(
            pod_id="p", key="k",
            deadline=time.time() - 1.0,  # already expired
            ssh_proc_ref=[proc],
            done_event=done,
            fired_event=fired,
            cleanup_event=cleanup,
            grace_sec=grace_sec,
            terminate_fn=terminate,
        )
        t.join(timeout=10.0)
        assert not t.is_alive()
        return order, fired

    def test_kills_ssh_before_any_delete_then_hard_deletes_after_grace(self, rl) -> None:
        order, fired = self._run(rl, proc=_FakeProc, cleanup_preset=False,
                                 grace_sec=0.05)
        assert order == ["kill", "terminate"]
        assert fired.is_set()

    def test_no_delete_when_main_thread_cleans_up_in_grace(self, rl) -> None:
        order, fired = self._run(rl, proc=_FakeProc, cleanup_preset=True,
                                 grace_sec=30.0)
        assert order == ["kill"]  # pod left alive for the artifact pull
        assert fired.is_set()

    def test_failed_kill_still_reaches_grace_delete(self, rl) -> None:
        order, fired = self._run(
            rl, proc=lambda o: _FakeProc(o, fail_kill=True),
            cleanup_preset=False, grace_sec=0.05,
        )
        assert order == ["terminate"]
        assert fired.is_set()

    def test_done_before_deadline_never_fires(self, rl) -> None:
        order, fired = self._run(rl, proc=_FakeProc, cleanup_preset=False,
                                 grace_sec=0.05, done_preset=True)
        assert order == []
        assert not fired.is_set()

    def test_terminate_pod_api_error_does_not_kill_watchdog_thread(
        self, rl, monkeypatch, capsys
    ) -> None:
        # End-to-end thread survival: the default terminate path hitting a
        # RunPodAPIError must not silently kill the daemon thread (the old
        # sys.exit-in-_req bug).
        def failing_req(method, path, key, body=None):  # noqa: ARG001
            raise rl.RunPodAPIError("RunPod API 500: boom")

        monkeypatch.setattr(rl, "_req", failing_req)
        done = threading.Event()
        fired = threading.Event()
        cleanup = threading.Event()
        t = rl._start_pod_watchdog(
            pod_id="p", key="k", deadline=time.time() - 1.0,
            ssh_proc_ref=[None], done_event=done, fired_event=fired,
            cleanup_event=cleanup, grace_sec=0.05,
        )
        t.join(timeout=10.0)
        assert not t.is_alive()
        assert fired.is_set()
        assert "POD TERMINATION FAILED" in capsys.readouterr().err
