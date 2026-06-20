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

Requires outbound access to rest.runpod.io. In a restricted/sandboxed network
environment this host may be blocked by the environment's network egress policy —
run this from a machine with RunPod access, or add rest.runpod.io to the allow-list.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REST = "https://rest.runpod.io/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

PIPELINE = r"""
set -euo pipefail
cd /workspace/pragmatiq
# Bound the CPU thread pools so the sequential pipeline stages — the
# gradient-boosting probe and the embedding pass especially — don't oversubscribe
# a many-core host and stall on thread-pool contention.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
pip install -q -e ".[dev,serve]"
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
    import json
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
               pubkey: str | None = None, min_vcpu: int = 8) -> dict:
    """Create an on-demand PyTorch pod with the requested GPU type(s).

    ``gpu`` may be a comma-separated preference list, e.g.
    "NVIDIA A100 80GB PCIe,NVIDIA A100 SXM 80GB". ``min_vcpu`` is plumbed to
    RunPod's ``minVCPUPerGPU`` filter (8 covers the 8-core throughput
    benchmark; higher counts speed up the CPU-bound tokenize stages).
    """
    body = {
        "name": name,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cloudType": cloud,
        "gpuTypeIds": [g.strip() for g in gpu.split(",") if g.strip()],
        "gpuCount": 1,
        "containerDiskInGb": 80,
        "volumeInGb": 100,
        "minVCPUPerGPU": min_vcpu,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
    }
    if pubkey:
        body["env"] = {"PUBLIC_KEY": pubkey, "SSH_PUBLIC_KEY": pubkey}
    return _req("POST", "/pods", key, body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB",
                    help="RunPod GPU type id(s), comma-separated by preference.")
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
    args = ap.parse_args()
    key = _api_key()

    if args.terminate:
        _req("DELETE", f"/pods/{args.terminate}", key)
        print(f"terminated pod {args.terminate}")
        return

    pub_file = _public_key_file(args.public_key_file)
    pubkey = Path(pub_file).read_text().strip() if pub_file else None
    identity = pub_file[:-4] if pub_file and pub_file.endswith(".pub") else None
    if pubkey is None:
        print("warning: no SSH public key found; relying on keys registered in the RunPod account")
    print(f"creating pod ({args.gpu}, {args.cloud_type}) ...")
    pod = create_pod(key, args.gpu, args.run_name, cloud=args.cloud_type, pubkey=pubkey,
                     min_vcpu=args.min_vcpu)
    pod_id = pod.get("id")
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
        sys.exit(f"pod {pod_id} did not expose SSH in time; check the RunPod console")
    ip, port = ssh
    print(f"pod ready at {ip}:{port}")

    id_opt = ["-i", identity] if identity else []
    if args.no_run:
        print(f"skip run; SSH: ssh {' '.join(id_opt)} root@{ip} -p {port}".replace("  ", " "))
        return

    # Sync the repo (no GitHub needed): write a local tarball, copy it as a file,
    # and verify the byte count survived before extracting. A streamed `tar -x`
    # over SSH can truncate on a community-host network blip, so we copy a file
    # and check its size rather than piping bytes through the connection.
    import tempfile

    ssh_base = ["ssh", "-o", "StrictHostKeyChecking=no", *id_opt, "-p", str(port), f"root@{ip}"]
    scp_base = ["scp", "-o", "StrictHostKeyChecking=no", *id_opt, "-P", str(port)]
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
            sys.exit(f"repo tarball truncated in transit ({remote_size} != {local_size} bytes); re-run")
        subprocess.run(ssh_base + [
            "tar -x -C /workspace/pragmatiq -f /workspace/pragmatiq.tar && rm /workspace/pragmatiq.tar"
        ], check=True)
    print("repo synced; running pipeline (this trains on the GPU) ...")
    subprocess.run(ssh_base + [PIPELINE], check=True)
    print(f"done. terminate with: python scripts/runpod_launch.py --terminate {pod_id}")


if __name__ == "__main__":
    main()
