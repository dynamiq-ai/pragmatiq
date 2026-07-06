#!/usr/bin/env python3
"""Launch pragmatiq on a RunPod GPU pod and run the end-to-end pipeline.

This is the turnkey path for validating pragmatiq on a real A100/H100/H200:
it creates a pod via the RunPod REST API, waits for SSH, syncs this repo
(no GitHub required — it tars and copies over SSH), installs, and runs the
GPU end-to-end: synth -> tokenize -> pretrain -> embed -> gradient-boosting probe,
plus the auto-config + gradient-accumulation path, the PRAGMA+Nemotron MSE variant,
the Triton serving contract, and the full-scale training and AML acceptance checks.

Usage:
    export RUNPOD_API_KEY=...            # or put it in .env (gitignored)
    python scripts/runpod_launch.py --gpu "NVIDIA A100 80GB PCIe" --run-name a100-smoke
    python scripts/runpod_launch.py --terminate <pod_id>
    python scripts/runpod_launch.py --dry-run --gpu-count 8 --gpu "NVIDIA H100 80GB HBM3" \\
        --cloud-type SECURE --remote-script scripts/validate_gpu.py --terminate-on-done

Requires outbound access to rest.runpod.io. In a restricted/sandboxed network
environment this host may be blocked by the environment's network egress policy —
run this from a machine with RunPod access, or add rest.runpod.io to the allow-list.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

REST = "https://rest.runpod.io/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Flash-attn prebuilt wheel for runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel
# torch==2.4.0, python==3.11, abi=FALSE.
# flash-attn 2.6.3 ships cu118 and cu123 wheels only (no cu124); the cu123
# wheel runs correctly on a cu124 runtime.  Using cu124 in the URL → 404.
# Source: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3
# ---------------------------------------------------------------------------
_FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

# Common install block executed on the pod before any command.
# - Installs [dev,train,serve] extras so Lightning and training deps are present.
# - Attempts to install the prebuilt flash-attn wheel; falls back to source build;
#   a failure is non-fatal (SDPA fallback exists in the model).
INSTALL = f"""
set -uo pipefail
cd /workspace/pragmatiq
pip install -q -e ".[dev,train,serve]"
echo "=== installing flash-attn ==="
pip install -q "{_FLASH_ATTN_WHEEL}" || {{
    echo "Prebuilt wheel not found; trying source build (slow)..."
    pip install flash-attn==2.6.3 --no-build-isolation -q || \
        echo "WARNING: flash-attn install failed; SDPA fallback will be used"
}}
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
python -c "import flash_attn; print('flash', flash_attn.__version__)" || \
    echo "flash-attn unavailable -> SDPA"
""".strip()

PIPELINE = r"""
set -euo pipefail
cd /workspace/pragmatiq
# Bound the CPU thread pools so the sequential pipeline stages — the
# gradient-boosting probe and the embedding pass especially — don't oversubscribe
# a many-core host and stall on thread-pool contention.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
python -X faulthandler -u - <<'PY'
from pragmatiq import api
m = api.synthesize({"n_users": 50000, "seed": 0}, out="data/synth", n_workers=8, write_report=True)
print("synth:", m["n_events"], "events", m["users_per_sec"], "users/s")
api.tokenize("data/synth", "data/tok")
s = api.pretrain("data/tok", "gpu-smoke", model_size="small",
                 config={"max_steps": 4000, "token_budget": 32768})
print("pretrain:", s["last_metrics"])
print("embed:", api.embed("data/tok", s["run_dir"], out="embeddings.parquet"))
# gradient-boosting probe (default) — reports ROC-AUC + PR-AUC vs the same-classifier baseline
print("probe:", api.probe("data/tok", s["run_dir"], "data/synth/labels/default_12m.parquet"))

# WF-3 scale knobs: auto-config sizes token_budget / grad_accum / schedule from the
# data + this GPU; an explicit max_steps keeps the smoke short.
sa = api.pretrain("data/tok", "gpu-auto", model_size="small", config="auto",
                  max_steps=300, grad_accum_steps=2)
print("auto-config + grad-accum pretrain:", sa["last_metrics"])

# PRAGMA+Nemotron variant: embed-mode tokenization auto-wires the MSE text branch.
# The `hash` stand-in keeps this leg fast; for the real embedder install ".[text]"
# and set text_encoder="nemotron" (text_encoder_dim is read from the model).
api.tokenize("data/synth", "data/tok_embed",
             config={"text_value_mode": "embed", "text_encoder": "hash"})
sn = api.pretrain("data/tok_embed", "gpu-nemo", model_size="small",
                  config={"max_steps": 1000, "token_budget": 16384})
print("nemotron-variant pretrain:", sn["last_metrics"])  # carries loss_text_mse
print("nemotron probe:", api.probe("data/tok_embed", sn["run_dir"],
                                    "data/synth/labels/default_12m.parquet"))
PY

# Serving contract on the production path (no Docker needed): the Triton model.py
# request->response cycle on GPU. Full container serving: scripts/deploy_serving.sh.
python -m pytest tests/test_inference.py::TestTritonServingContract -q

# Full-scale acceptance checks (the quality bar; flash≡SDPA, probe>baseline, AML recovery)
PRAGMATIQ_GATE_FULL=1 PRAGMATIQ_GATE_SKIP_UNIT=1 bash scripts/gates/gate_5.sh
PRAGMATIQ_GATE_FULL=1 bash scripts/gates/gate_6.sh
"""


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("RUNPOD_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("set RUNPOD_API_KEY (env or .env)")
    return key


class RunPodAPIError(RuntimeError):
    """A RunPod REST call failed (HTTP error or unreachable host).

    Raised instead of ``sys.exit`` so that cleanup code — the watchdog thread
    and ``_terminate_pod`` in particular — can catch it as a plain Exception.
    ``SystemExit`` does NOT inherit from Exception, so an exiting ``_req``
    would silently kill the watchdog daemon thread mid-cleanup and leak a
    paid pod. Top-level CLI call sites convert this to ``sys.exit``.
    """


def _req(method: str, path: str, key: str, body: dict | None = None) -> dict:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{REST}{path}", method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode().strip()
            return json.loads(raw) if raw else {}  # DELETE returns an empty body
    except urllib.error.HTTPError as e:  # surface the deny reason clearly
        raise RunPodAPIError(f"RunPod API {e.code}: {e.read().decode()[:300]}") from e
    except urllib.error.URLError as e:
        raise RunPodAPIError(
            f"cannot reach {REST} ({e.reason}). Egress to rest.runpod.io may be blocked."
        ) from e


def _wait_for_ssh(
    pod_id: str,
    key: str,
    *,
    attempts: int = 60,
    interval: float = 10.0,
    req: Callable[..., dict] = _req,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, int] | None:
    """Poll the pod until it exposes SSH; return ``(ip, port)`` or None on timeout.

    Transient API failures (rate limits, blips) consume an attempt with a
    backed-off sleep instead of aborting: aborting here used to leak the pod
    because termination cleanup was only armed after this loop succeeded.
    """
    consecutive_failures = 0
    for _ in range(attempts):
        try:
            info = req("GET", f"/pods/{pod_id}", key)
        except RunPodAPIError as exc:
            consecutive_failures += 1
            print(f"[poll] transient API failure ({exc}); retrying ...", file=sys.stderr)
            sleep(min(interval * (2 ** min(consecutive_failures, 4)), 120.0))
            continue
        consecutive_failures = 0
        ports = info.get("portMappings") or {}
        ip = info.get("publicIp")
        if ip and "22" in {str(k) for k in ports}:
            return ip, int(ports["22"])
        sleep(interval)
    return None


def _public_key_file(path: str | None) -> str | None:
    """Locate an SSH public key to inject into the pod (PUBLIC_KEY env).

    Without this, SSH only works if the RunPod account already has a key
    registered in console settings. Auto-detects common key names if no
    path is given; returns the path, or None if nothing is found (account
    keys may still work).
    """
    candidates = [Path(path).expanduser()] if path else [
        Path.home() / ".ssh" / n
        for n in ("runpod_pragmatiq_ed25519.pub", "id_ed25519.pub", "id_rsa.pub")
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def create_pod(key: str, gpu: str, name: str, cloud: str = "COMMUNITY",
               pubkey: str | None = None, min_vcpu: int = 8,
               gpu_count: int = 1) -> dict:
    """Create an on-demand PyTorch pod with the requested GPU type(s).

    ``gpu`` may be a comma-separated preference list, e.g.
    "NVIDIA A100 80GB PCIe,NVIDIA A100 SXM 80GB". ``min_vcpu`` is plumbed to
    RunPod's ``minVCPUPerGPU`` filter (8 covers the 8-core throughput
    benchmark; higher counts speed up the CPU-bound tokenize stages).
    ``gpu_count`` sets the number of GPUs per pod; disk is scaled up for
    large-model checkpoints + flash-attn when gpu_count is large.
    """
    # Scale container disk with GPU count (checkpoints + flash-attn cache).
    # Cap at 500 GB to stay within typical RunPod limits.
    container_disk = min(max(80, 30 * gpu_count), 500)

    # NOTE on /dev/shm size: The RunPod REST v1 pod-create body does NOT expose
    # a shmSize or equivalent field (unlike docker run --shm-size).  The container
    # gets whatever RunPod provisions by default (typically 64 MB), which is too
    # small for NCCL shared-memory transport on multi-GPU pods and causes the
    # NCCLUtils.hpp ncclUnhandledCudaError at d>=4 init.  The harness works around
    # this by setting NCCL_SHM_DISABLE=1 + NCCL_P2P_DISABLE=1 + NCCL_IB_DISABLE=1
    # in the DDP leg subprocess env (--nccl-safe=on, default).  If RunPod ever
    # adds a shmSize knob to their REST API, set it here (e.g., "shmSizeGb": 8)
    # and remove the NCCL_SHM_DISABLE env var to get NVLink-optimal throughput.
    body = {
        "name": name,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cloudType": cloud,
        "gpuTypeIds": [g.strip() for g in gpu.split(",") if g.strip()],
        "gpuCount": gpu_count,
        "containerDiskInGb": container_disk,
        "volumeInGb": 100,
        "minVCPUPerGPU": min_vcpu,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
    }
    if pubkey:
        body["env"] = {"PUBLIC_KEY": pubkey, "SSH_PUBLIC_KEY": pubkey}
    return _req("POST", "/pods", key, body)


def _terminate_pod(pod_id: str, key: str) -> None:
    """Terminate a pod, logging the outcome. Errors are surfaced but not raised.

    On ANY failure (API unreachable, HTTP error, network timeout) a loud warning
    is printed to stderr so the operator can clean up manually.  The exception is
    swallowed so it cannot mask an original exception in the caller's finally block.
    """
    try:
        _req("DELETE", f"/pods/{pod_id}", key)
        print(f"[terminate] pod {pod_id} terminated.")
    except Exception as exc:  # noqa: BLE001 — must not propagate; loud warning instead
        print(
            f"\n!!! POD TERMINATION FAILED for {pod_id} — "
            f"MANUALLY TERMINATE: python scripts/runpod_launch.py --terminate {pod_id} !!!\n"
            f"    (error: {exc})",
            file=sys.stderr,
        )


def _start_pod_watchdog(
    pod_id: str,
    key: str,
    deadline: float,
    ssh_proc_ref: list[subprocess.Popen | None],
    done_event: threading.Event,
    fired_event: threading.Event,
    cleanup_event: threading.Event,
    grace_sec: float = 600.0,
    terminate_fn: Callable[[str, str], None] | None = None,
) -> threading.Thread:
    """Start a daemon watchdog thread that enforces the runtime cap at *deadline*.

    The watchdog is independent of the SSH command subprocess: even if the SSH
    connection is wedged (e.g., the remote harness is deadlocked), the cap
    fires at ``deadline`` (= ``create_ts + max_runtime_min * 60``).

    On deadline the watchdog kills the SSH subprocess FIRST (in its own try,
    so a later API failure can never leave the launcher blocked on SSH) and
    sets ``fired_event``; it does NOT delete the pod yet — the main thread's
    finally block pulls artifacts from the still-alive pod and then terminates
    it, so a paid run's outputs survive the cap. Only if the main thread has
    not signalled ``cleanup_event`` within ``grace_sec`` (pull hung, launcher
    wedged) does the watchdog hard-DELETE the pod itself.

    Args:
        pod_id:        RunPod pod ID to DELETE at the grace deadline.
        key:           RunPod API key.
        deadline:      ``time.time()`` timestamp at which the cap fires.
        ssh_proc_ref:  Single-element list; element 0 is the SSH subprocess (or
                       None before it starts).
        done_event:    Set by the caller when the run finishes normally; causes
                       the watchdog to exit without firing.
        fired_event:   Set by the watchdog when the cap fires; the main thread
                       uses it to force termination in its finally block.
        cleanup_event: Set by the main thread once its finally block has
                       terminated the pod; disarms the grace hard-DELETE.
        grace_sec:     Seconds after firing before the hard-DELETE fallback.
        terminate_fn:  Injectable for tests; defaults to ``_terminate_pod``.

    Returns:
        The started daemon thread (already running; join it after cleanup).
    """
    terminate = terminate_fn if terminate_fn is not None else _terminate_pod

    def _watch() -> None:
        remaining = deadline - time.time()
        if remaining > 0:
            # Wait until the deadline OR until the run finishes normally.
            done_event.wait(timeout=remaining)

        if done_event.is_set():
            # Run finished cleanly before the deadline — nothing to do.
            return

        # ---- WATCHDOG FIRES ----
        fired_event.set()
        # Kill SSH first, in its own try: unblocking the main thread must not
        # depend on any API call succeeding.
        proc = ssh_proc_ref[0]
        if proc is not None:
            try:
                proc.kill()
                print("\n[WATCHDOG] deadline reached; SSH subprocess killed — "
                      "main thread will pull artifacts, then terminate the pod.",
                      file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"\n[WATCHDOG] could not kill SSH subprocess: {exc}",
                      file=sys.stderr, flush=True)
        else:
            print("\n[WATCHDOG] deadline reached before the SSH command started.",
                  file=sys.stderr, flush=True)

        # Grace period: give the main thread time to pull artifacts and
        # terminate the pod itself; hard-DELETE only if it never does.
        if cleanup_event.wait(timeout=grace_sec):
            return
        print(f"[WATCHDOG] cleanup did not finish within {grace_sec:.0f}s grace; "
              f"hard-terminating pod {pod_id} ...", file=sys.stderr, flush=True)
        terminate(pod_id, key)

    t = threading.Thread(target=_watch, name="pod-watchdog", daemon=True)
    t.start()
    return t


# Bulk paths excluded from the artifact pull by default: checkpoints and
# datasets run to tens of GB on real GPU runs and blow past any sane timeout,
# leaving a truncated (corrupt) archive. The reports, metrics CSVs, and logs
# are what the operator actually needs after the pod dies.
_DEFAULT_PULL_EXCLUDES: tuple[str, ...] = (
    "*/checkpoints",
    "checkpoints",
    "*/data",
    "data",
    "*.ckpt",
    "*.pt",
    "*.safetensors",
)


def _build_pull_cmd(
    pull_glob: str, excludes: Sequence[str] = _DEFAULT_PULL_EXCLUDES
) -> str:
    """Build the remote tar-over-ssh command that streams matching artifacts.

    The remote shell expands ``pull_glob``; ``--exclude`` patterns must precede
    the file operands for GNU tar to apply them during archive creation.
    ``2>/dev/null`` suppresses "no match" noise from the remote shell.
    """
    exclude_opts = " ".join(f"--exclude='{p}'" for p in excludes)
    tar_cmd = f"tar czf - {exclude_opts} {pull_glob}".replace("  ", " ")
    return f"cd /workspace/pragmatiq && {tar_cmd} 2>/dev/null"


def _pull_artifacts(
    ssh_base: list[str],
    pull_glob: str,
    pull_dest: str,
    *,
    timeout_sec: float = 1800.0,
    excludes: Sequence[str] = _DEFAULT_PULL_EXCLUDES,
) -> bool:
    """Best-effort: pull remote artifacts back to the local machine via tar-over-ssh.

    ``scp -r`` does NOT expand a remote glob when the path contains shell
    metacharacters — the literal string is passed to the server and silently
    pulls nothing.  Instead we stream a tar archive over SSH so the remote shell
    expands the glob, then extract it locally.  The archive is streamed to
    ``pulled.tar.gz.part`` and renamed to ``<pull_dest>/pulled.tar.gz`` only on
    success, so a timeout or dropped connection can never leave a truncated
    archive masquerading as a complete one; the partial file is removed.

    Failures are logged as warnings and do not raise, so cleanup in the caller's
    finally block always proceeds. Returns True iff the pull succeeded.
    """
    local_dest = Path(pull_dest)
    local_dest.mkdir(parents=True, exist_ok=True)
    archive = local_dest / "pulled.tar.gz"
    partial = archive.with_name(archive.name + ".part")
    remote_cmd = _build_pull_cmd(pull_glob, excludes)
    try:
        with partial.open("wb") as fh:
            subprocess.run(
                ssh_base + [remote_cmd],
                stdout=fh,
                check=True,
                timeout=timeout_sec,
            )
        # Zero-byte archive means the glob matched nothing — treat as warning.
        if partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            print(f"[pull] WARNING: no files matched glob '{pull_glob}' on the pod",
                  file=sys.stderr)
            return False
        partial.replace(archive)
        # Extract in place so individual files are available alongside the archive.
        subprocess.run(
            ["tar", "xzf", str(archive), "-C", str(local_dest)],
            check=True, timeout=timeout_sec,
        )
        print(f"[pull] artifacts pulled (glob '{pull_glob}') -> {local_dest}")
        return True
    except Exception as exc:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        print(f"[pull] WARNING: could not pull glob '{pull_glob}': {exc}", file=sys.stderr)
        return False


def main() -> None:  # noqa: C901 — long but linear; split would obscure flow
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB",
                    help="RunPod GPU type id(s), comma-separated by preference.")
    ap.add_argument("--gpu-count", type=int, default=1, metavar="N",
                    help="Number of GPUs per pod (default 1; use 8 for 8×H100).")
    ap.add_argument("--cloud-type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"],
                    help="COMMUNITY is cheaper; SECURE has vetted datacenter hosts.")
    ap.add_argument("--public-key-file", default=None,
                    help="SSH public key to inject into the pod (default: auto-detect ~/.ssh).")
    ap.add_argument("--min-vcpu", type=int, default=8,
                    help="Minimum vCPUs per GPU (RunPod minVCPUPerGPU); 8 matches the "
                         "throughput benchmark, higher speeds CPU-bound tokenize stages.")
    ap.add_argument("--run-name", default="pragmatiq-smoke")
    ap.add_argument("--terminate", metavar="POD_ID", help="Terminate a pod and exit.")
    ap.add_argument("--no-run", action="store_true", help="Create the pod but don't run the pipeline.")
    # Remote script / harness
    ap.add_argument("--remote-script", default=None, metavar="PATH",
                    help="Local path to a script to upload and run on the pod instead of "
                         "the default PIPELINE (e.g. scripts/validate_gpu.py). "
                         "The repo is synced to /workspace/pragmatiq so the script is "
                         "available at /workspace/pragmatiq/<PATH>.")
    ap.add_argument("--remote-args", default="", metavar="ARGS",
                    help="Extra arguments to pass to --remote-script (quoted string).")
    # Artifact pull-back
    ap.add_argument("--pull", default="outputs/gpu-validation-*", metavar="GLOB",
                    help="Remote glob (relative to /workspace/pragmatiq) to pull back "
                         "after the run (default: outputs/gpu-validation-*).")
    ap.add_argument("--pull-dest", default=".", metavar="DIR",
                    help="Local directory to write pulled artifacts (default: current dir).")
    ap.add_argument("--pull-timeout-sec", type=float, default=1800.0, metavar="SEC",
                    help="Timeout for the artifact pull-back stream (default 1800). "
                         "Raise it when pulling large artifacts over slow links.")
    ap.add_argument("--pull-exclude", action="append", default=None, metavar="PATTERN",
                    help="tar --exclude pattern applied to the artifact pull; repeatable. "
                         f"Default: {' '.join(_DEFAULT_PULL_EXCLUDES)} (checkpoints and "
                         "datasets are tens of GB and would truncate the pull).")
    # Auto-terminate / safety
    ap.add_argument("--terminate-on-done", action="store_true",
                    help="DELETE the pod on every exit path: success, exception, timeout, "
                         "or Ctrl-C. CRITICAL for cost control (~$30-40/hr for 8×H100).")
    ap.add_argument("--max-runtime-min", type=int, default=0, metavar="N",
                    help="Hard wall-clock timeout in minutes (0 = unlimited). "
                         "On expiry, artifacts are pulled and the pod is terminated.")
    ap.add_argument("--usd-per-hour", type=float, default=35.0, metavar="RATE",
                    help="Hourly cost estimate for the pod (default 35 for 8×H100); "
                         "used only for the informational cost printout.")
    # Dry-run
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the resolved create-pod body, INSTALL block, and command "
                         "that WOULD run, then exit without calling RunPod.")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Standalone terminate mode (existing behaviour, unchanged)
    # ------------------------------------------------------------------
    if args.terminate:
        key = _api_key()
        try:
            _req("DELETE", f"/pods/{args.terminate}", key)
        except RunPodAPIError as exc:
            sys.exit(str(exc))
        print(f"terminated pod {args.terminate}")
        return

    # ------------------------------------------------------------------
    # Build resolved pod body and command for display / dry-run
    # ------------------------------------------------------------------
    gpu_count = args.gpu_count
    container_disk = min(max(80, 30 * gpu_count), 500)
    pod_body_preview = {
        "name": args.run_name,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cloudType": args.cloud_type,
        "gpuTypeIds": [g.strip() for g in args.gpu.split(",") if g.strip()],
        "gpuCount": gpu_count,
        "containerDiskInGb": container_disk,
        "volumeInGb": 100,
        "minVCPUPerGPU": args.min_vcpu,
    }

    if args.remote_script:
        remote_path = f"/workspace/pragmatiq/{args.remote_script}"
        extra = f" {args.remote_args}" if args.remote_args.strip() else ""
        run_command = f"python -X faulthandler -u {remote_path}{extra}"
    else:
        run_command = "(default PIPELINE)"

    # ------------------------------------------------------------------
    # --dry-run: print and exit without touching the network
    # ------------------------------------------------------------------
    if args.dry_run:
        print("=== DRY RUN — no network calls will be made ===\n")
        print("--- resolved create_pod body ---")
        print(json.dumps(pod_body_preview, indent=2))
        print("\n--- INSTALL block ---")
        print(INSTALL)
        print("\n--- command that would run ---")
        print(run_command)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Live run — require API key only here
    # ------------------------------------------------------------------
    key = _api_key()

    pub_file = _public_key_file(args.public_key_file)
    pubkey = Path(pub_file).read_text().strip() if pub_file else None
    identity = pub_file[:-4] if pub_file and pub_file.endswith(".pub") else None
    if pubkey is None:
        print("warning: no SSH public key found; relying on keys registered in the RunPod account")

    # ------------------------------------------------------------------
    # Install SIGINT/SIGTERM handlers BEFORE creating the pod: a Ctrl-C /
    # kill that lands between pod creation and the try/finally below must
    # still reach the cleanup path, or --terminate-on-done leaks a paid pod.
    # ------------------------------------------------------------------
    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"\n[signal] received signal {signum}; cleaning up ...", file=sys.stderr)
        # Raise KeyboardInterrupt so the try/finally fires.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"creating pod ({args.gpu} ×{gpu_count}, {args.cloud_type}) ...")
    try:
        pod = create_pod(key, args.gpu, args.run_name, cloud=args.cloud_type, pubkey=pubkey,
                         min_vcpu=args.min_vcpu, gpu_count=gpu_count)
    except RunPodAPIError as exc:
        sys.exit(str(exc))
    pod_id = pod.get("id")
    if not pod_id:
        sys.exit(f"pod create returned no id: {json.dumps(pod)[:300]}")
    # Record the creation timestamp: billing starts here, so both the runtime
    # cap and the cost printout are measured from it (not from SSH-ready).
    create_ts = time.time()
    print(f"pod {pod_id} created; polling for SSH ...")

    # Exit-code tracking: stays 0 for a clean run; set to 1 on SSH non-zero,
    # watchdog kill, or any unhandled exception.  Pod termination in the
    # finally block is UNCONDITIONAL — the exit code is propagated only AFTER
    # cleanup so a failed pod run doesn't leak a paid instance.
    _exit_code = 0
    run_started = False
    ssh_base: list[str] | None = None
    _watchdog_thread: threading.Thread | None = None
    _ssh_proc_ref: list[subprocess.Popen | None] = [None]
    _run_done = threading.Event()
    _watchdog_fired = threading.Event()
    _cleanup_done = threading.Event()
    pull_excludes: tuple[str, ...] = (
        tuple(args.pull_exclude) if args.pull_exclude else _DEFAULT_PULL_EXCLUDES
    )

    # The pod exists and is billing from this point: EVERY path below —
    # SSH-poll failure, sync failure, Ctrl-C, watchdog deadline — must flow
    # through the finally block so --terminate-on-done can clean up.
    try:
        # ---- watchdog: armed from creation, covers polling + sync too ----
        # The watchdog fires at create_ts + max_runtime_min*60 independently
        # of the SSH subprocess. On firing it kills SSH and sets
        # _watchdog_fired; the finally block pulls artifacts from the
        # still-alive pod and then terminates it. Only if this cleanup stalls
        # past the grace window does the watchdog hard-DELETE the pod itself.
        if args.max_runtime_min > 0:
            watchdog_deadline = create_ts + args.max_runtime_min * 60
            print(
                f"[watchdog] armed — runtime cap in ~{args.max_runtime_min:.1f} min "
                "(measured from pod creation); artifacts are pulled before termination",
                flush=True,
            )
            _watchdog_thread = _start_pod_watchdog(
                pod_id=pod_id,
                key=key,
                deadline=watchdog_deadline,
                ssh_proc_ref=_ssh_proc_ref,
                done_event=_run_done,
                fired_event=_watchdog_fired,
                cleanup_event=_cleanup_done,
                # The grace window must outlast the artifact pull, or the
                # hard-DELETE would kill the pod mid-pull.
                grace_sec=args.pull_timeout_sec + 120.0,
            )

        ssh = _wait_for_ssh(pod_id, key)
        if ssh is None:
            raise RuntimeError(
                f"pod {pod_id} did not expose SSH in time; check the RunPod console"
            )
        # The cap can fire while we were still polling for SSH (no subprocess
        # existed for the watchdog to kill) — starting the paid remote command
        # after the deadline would run it for the whole grace window for nothing.
        if _watchdog_fired.is_set():
            raise RuntimeError(
                f"runtime cap expired while waiting for pod {pod_id} SSH; "
                "not starting the remote command"
            )
        ip, port = ssh
        print(f"pod ready at {ip}:{port}")

        # Write a gitignored sidecar so external monitoring can find the pod
        # even when this launcher's stdout is buffered.  Best-effort: never fatal.
        _sidecar = REPO_ROOT / ".runpod_last.json"
        try:
            _sidecar_data = {
                "pod_id": pod_id,
                "ip": ip,
                "ssh_port": port,
                "created": create_ts,
            }
            _sidecar.write_text(json.dumps(_sidecar_data, indent=2))
            print(f"[pod-info] sidecar written: {_sidecar}")
            print(f"[pod-info] {json.dumps(_sidecar_data)}")
        except Exception as _sidecar_exc:  # noqa: BLE001
            print(f"[pod-info] WARNING: could not write sidecar: {_sidecar_exc}",
                  file=sys.stderr)

        id_opt = ["-i", identity] if identity else []
        if args.no_run:
            print(f"skip run; SSH: ssh {' '.join(id_opt)} root@{ip} -p {port}"
                  .replace("  ", " "))
            # The finally block handles --terminate-on-done; nothing to pull.
            return

        ssh_base = ["ssh", "-o", "StrictHostKeyChecking=no", *id_opt,
                    "-p", str(port), f"root@{ip}"]
        scp_base = ["scp", "-o", "StrictHostKeyChecking=no", *id_opt, "-P", str(port)]

        # ---- sync repo ------------------------------------------------
        run_started = True
        with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
            subprocess.run(["git", "archive", "--format=tar", "-o", tf.name, "HEAD"],
                           cwd=REPO_ROOT, check=True)
            local_size = Path(tf.name).stat().st_size
            subprocess.run(ssh_base + ["mkdir -p /workspace/pragmatiq"], check=True)
            subprocess.run(scp_base + [tf.name, f"root@{ip}:/workspace/pragmatiq.tar"], check=True)
            remote_size = int(subprocess.run(
                ssh_base + ["stat -c %s /workspace/pragmatiq.tar"],
                capture_output=True, text=True, check=True).stdout.strip())
            if remote_size != local_size:
                raise RuntimeError(
                    f"repo tarball truncated in transit ({remote_size} != {local_size} bytes)"
                )
            subprocess.run(ssh_base + [
                "tar -x -C /workspace/pragmatiq -f /workspace/pragmatiq.tar "
                "&& rm /workspace/pragmatiq.tar"
            ], check=True)
        print("repo synced; running install + command ...")

        # ---- build the on-pod shell command ---------------------------
        if args.remote_script:
            remote_path = f"/workspace/pragmatiq/{args.remote_script}"
            extra = f" {args.remote_args}" if args.remote_args.strip() else ""
            run_cmd = f"python -X faulthandler -u {remote_path}{extra}"
        else:
            run_cmd = PIPELINE

        # INSTALL drops errexit on purpose (the flash-attn fallback chain must
        # be allowed to fail); restore it and prove the editable install
        # actually works before spending GPU time on the real command.
        on_pod = (
            f"set -euo pipefail\n{INSTALL}\n"
            "set -e\n"
            'python -c "import pragmatiq; print(\'pragmatiq\', pragmatiq.__version__)"\n'
            f"{run_cmd}"
        )

        # ---- run on pod (hard deadline enforced by the watchdog) -------
        try:
            ssh_cmd = ssh_base + [on_pod]
            ssh_proc = subprocess.Popen(ssh_cmd)  # noqa: S603
            _ssh_proc_ref[0] = ssh_proc
            # Wait without a Python-level timeout; the watchdog provides the
            # hard deadline via proc.kill().
            ssh_proc.wait()
            if ssh_proc.returncode != 0:
                # rc=-9 indicates a watchdog kill; any non-zero rc is a failure.
                _exit_code = 1
                print(
                    f"[run] SSH command exited with rc={ssh_proc.returncode}",
                    file=sys.stderr,
                )
        finally:
            # Signal the watchdog that the run is done so it doesn't fire.
            # (If it already fired it is waiting on _cleanup_done instead,
            # which the outer finally sets after terminating the pod.)
            _run_done.set()

        if _watchdog_fired.is_set():
            _exit_code = 1
            print("[run] runtime cap hit; pulling artifacts before termination ...",
                  file=sys.stderr)
        elif _exit_code == 0:
            print(f"run complete (pod {pod_id})")

    except KeyboardInterrupt:
        print("[interrupt] KeyboardInterrupt caught; running cleanup ...", file=sys.stderr)
        _exit_code = 1
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        _exit_code = 1
    finally:
        # ---- pull artifacts BEFORE terminating (best-effort) ---------
        # Ordering is deliberate: on a watchdog deadline the pod is still
        # alive here, so the paid run's artifacts survive the cap.
        if run_started and ssh_base is not None:
            _pull_artifacts(ssh_base, args.pull, args.pull_dest,
                            timeout_sec=args.pull_timeout_sec, excludes=pull_excludes)

        # ---- cost estimate (billing starts at pod creation) ----------
        elapsed = time.time() - create_ts
        hours = elapsed / 3600
        cost = hours * args.usd_per_hour
        print(f"[cost] pod lifetime {elapsed / 60:.1f} min since creation; "
              f"estimated cost ~${cost:.2f} at ${args.usd_per_hour:.0f}/hr")

        # ---- auto-terminate (safety-critical) -----------------------
        # A fired watchdog forces termination even without --terminate-on-done:
        # --max-runtime-min promises the pod dies at the cap.
        if args.terminate_on_done or _watchdog_fired.is_set():
            _terminate_pod(pod_id, key)
        else:
            print(f"[info] pod still running; terminate with: "
                  f"python scripts/runpod_launch.py --terminate {pod_id}")

        # Disarm the watchdog's grace-period hard-DELETE and reap the thread.
        _run_done.set()
        _cleanup_done.set()
        if _watchdog_thread is not None:
            _watchdog_thread.join(timeout=5)

    # Propagate the remote result to the caller AFTER cleanup is done.
    if _exit_code:
        print(f"[main] exit={_exit_code} (remote run failed)", flush=True)
        sys.exit(_exit_code)


if __name__ == "__main__":
    main()
