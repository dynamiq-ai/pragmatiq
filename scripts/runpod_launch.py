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
import time
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


def _req(method: str, path: str, key: str, body: dict | None = None) -> dict:
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
        sys.exit(f"RunPod API {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {REST} ({e.reason}). Egress to rest.runpod.io may be blocked.")


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


def _pull_artifacts(
    ssh_base: list[str],
    scp_base: list[str],  # kept for signature compatibility; not used (tar-over-ssh)
    ip: str,
    port: int,
    pull_glob: str,
    pull_dest: str,
) -> None:
    """Best-effort: pull remote artifacts back to the local machine via tar-over-ssh.

    ``scp -r`` does NOT expand a remote glob when the path contains shell
    metacharacters — the literal string is passed to the server and silently
    pulls nothing.  Instead we stream a tar archive over SSH so the remote shell
    expands the glob, then extract it locally.  The resulting archive is written
    to ``<pull_dest>/pulled.tar.gz``.

    Failures are logged as warnings and do not raise, so partial results survive
    even when the pod terminates early.
    """
    local_dest = Path(pull_dest)
    local_dest.mkdir(parents=True, exist_ok=True)
    archive = local_dest / "pulled.tar.gz"
    # The remote shell expands the glob; 2>/dev/null suppresses "no match" noise.
    remote_cmd = f"cd /workspace/pragmatiq && tar czf - {pull_glob} 2>/dev/null"
    try:
        with archive.open("wb") as fh:
            subprocess.run(
                ssh_base + [remote_cmd],
                stdout=fh,
                check=True,
                timeout=300,
            )
        # Zero-byte archive means the glob matched nothing — treat as warning.
        if archive.stat().st_size == 0:
            archive.unlink(missing_ok=True)
            print(f"[pull] WARNING: no files matched glob '{pull_glob}' on the pod",
                  file=sys.stderr)
            return
        # Extract in place so individual files are available alongside the archive.
        subprocess.run(
            ["tar", "xzf", str(archive), "-C", str(local_dest)],
            check=True, timeout=120,
        )
        print(f"[pull] artifacts pulled (glob '{pull_glob}') -> {local_dest}")
    except Exception as exc:  # noqa: BLE001
        print(f"[pull] WARNING: could not pull glob '{pull_glob}': {exc}", file=sys.stderr)


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
        _req("DELETE", f"/pods/{args.terminate}", key)
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

    print(f"creating pod ({args.gpu} ×{gpu_count}, {args.cloud_type}) ...")
    pod = create_pod(key, args.gpu, args.run_name, cloud=args.cloud_type, pubkey=pubkey,
                     min_vcpu=args.min_vcpu, gpu_count=gpu_count)
    pod_id = pod.get("id")
    # Fix B: record creation timestamp so that sync+install time counts against
    # the wall-clock budget, not just command execution time.
    create_ts = time.time()
    print(f"pod {pod_id} created; polling for SSH ...")

    ssh = None
    for _ in range(60):
        info = _req("GET", f"/pods/{pod_id}", key)
        ports = info.get("portMappings") or {}
        ip = info.get("publicIp")
        if ip and "22" in {str(k) for k in ports}:
            ssh = (ip, int(ports["22"]))
            break
        time.sleep(10)
    if ssh is None:
        if args.terminate_on_done:
            _terminate_pod(pod_id, key)
        sys.exit(f"pod {pod_id} did not expose SSH in time; check the RunPod console")
    ip, port = ssh
    print(f"pod ready at {ip}:{port}")

    # Fix D: write a gitignored sidecar so external monitoring can find the pod
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
        print(f"[pod-info] WARNING: could not write sidecar: {_sidecar_exc}", file=sys.stderr)

    id_opt = ["-i", identity] if identity else []
    if args.no_run:
        print(f"skip run; SSH: ssh {' '.join(id_opt)} root@{ip} -p {port}".replace("  ", " "))
        # CRITICAL: terminate before returning so --no-run --terminate-on-done
        # does not leak a paid pod.  The try/finally below is never entered on
        # this path, so termination must happen here explicitly.
        if args.terminate_on_done:
            _terminate_pod(pod_id, key)
        else:
            print(f"[info] pod still running; terminate with: "
                  f"python scripts/runpod_launch.py --terminate {pod_id}")
        return

    ssh_base = ["ssh", "-o", "StrictHostKeyChecking=no", *id_opt, "-p", str(port), f"root@{ip}"]
    scp_base = ["scp", "-o", "StrictHostKeyChecking=no", *id_opt, "-P", str(port)]

    # ------------------------------------------------------------------
    # Install SIGINT/SIGTERM handlers so Ctrl-C / kill → finally block
    # runs (and terminates the pod when --terminate-on-done).
    # ------------------------------------------------------------------
    _interrupted = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal _interrupted
        print(f"\n[signal] received signal {signum}; cleaning up ...", file=sys.stderr)
        _interrupted = True
        # Raise KeyboardInterrupt so the try/finally fires.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_time = time.monotonic()

    import tempfile

    try:
        # ---- sync repo ------------------------------------------------
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

        on_pod = f"set -euo pipefail\n{INSTALL}\n{run_cmd}"

        # ---- run on pod (with optional timeout) -----------------------
        # Fix B: compute remaining seconds from pod-creation so that sync +
        # install time counts against the wall-clock cap, not just this command.
        if args.max_runtime_min > 0:
            remaining_sec = max(60.0, create_ts + args.max_runtime_min * 60 - time.time())
        else:
            remaining_sec = None
        try:
            subprocess.run(
                ssh_base + [on_pod],
                check=True,
                timeout=remaining_sec,
            )
        except subprocess.TimeoutExpired:
            print(f"[timeout] max-runtime-min={args.max_runtime_min} wall-clock exceeded "
                  "(measured from pod creation); "
                  "pulling artifacts and terminating ...", file=sys.stderr)
        print(f"run complete (pod {pod_id})")

    except KeyboardInterrupt:
        print("[interrupt] KeyboardInterrupt caught; running cleanup ...", file=sys.stderr)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
    finally:
        # ---- pull artifacts (best-effort, always) --------------------
        _pull_artifacts(ssh_base, scp_base, ip, port, args.pull, args.pull_dest)

        # ---- cost estimate ------------------------------------------
        elapsed = time.monotonic() - start_time
        hours = elapsed / 3600
        cost = hours * args.usd_per_hour
        print(f"[cost] elapsed {elapsed / 60:.1f} min; "
              f"estimated cost ~${cost:.2f} at ${args.usd_per_hour:.0f}/hr")

        # ---- auto-terminate (safety-critical) -----------------------
        if args.terminate_on_done:
            _terminate_pod(pod_id, key)
        else:
            print(f"[info] pod still running; terminate with: "
                  f"python scripts/runpod_launch.py --terminate {pod_id}")


if __name__ == "__main__":
    main()
