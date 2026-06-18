"""Typer CLI — a thin wrapper: parse args, call :mod:`pragmatiq.api`, print.

No logic lives here (global rule 1). Command results print to stdout;
progress and log output go to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(name="pragmatiq", help="pragmatiq: behavioral foundation model for banking events.",
                  no_args_is_help=True, pretty_exceptions_show_locals=False)
synth_app = typer.Typer(help="Synthetic data generation.", no_args_is_help=True)
app.add_typer(synth_app, name="synth")


@app.callback()
def _setup(
    ctx: typer.Context,
    verbose: bool = typer.Option(True, "--verbose/--quiet",
                                 help="Show INFO-level progress logs on stderr."),
) -> None:
    ctx.obj = {"verbose": verbose}
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        stream=sys.stderr, format="%(message)s")


@synth_app.command("generate")
def synth_generate(
    out: Path = typer.Option(Path("data/synth"), help="Output directory."),
    config: Path | None = typer.Option(None, help="WorldConfig YAML (configs/data/synthetic.yaml)."),
    n_users: int | None = typer.Option(None, help="Override n_users."),
    seed: int | None = typer.Option(None, help="Override seed."),
    n_workers: int = typer.Option(0, help="Parallel workers (<=1 = inline)."),
    report: bool = typer.Option(True, help="Write realism_report.html."),
) -> None:
    """Generate a synthetic banking dataset."""
    from pragmatiq import api

    manifest = api.synthesize(config=config, out=out, n_users=n_users, seed=seed,
                              n_workers=n_workers, write_report=report)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("tokenize")
def tokenize_cmd(
    data_dir: Path = typer.Argument(..., help="Generated dataset directory."),
    out: Path = typer.Option(Path("data/tokenized"), help="Output (shards + tokenizer + index)."),
    config: Path | None = typer.Option(None, help="Tokenizer YAML (configs/data/tokenizer.yaml)."),
    tokenizer_dir: Path | None = typer.Option(None, help="Reuse an existing tokenizer dir."),
    max_users: int | None = typer.Option(None, help="Cap users (debug)."),
    n_workers: int = typer.Option(0, help="Parallel encode workers (<=1 = inline)."),
) -> None:
    """Fit/apply the tokenizer and write tokenized shards + LMDB index."""
    from pragmatiq import api

    manifest = api.tokenize(data_dir=data_dir, out=out, config=config,
                            tokenizer_dir=tokenizer_dir, max_users=max_users,
                            n_workers=n_workers)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("pretrain")
def pretrain_cmd(
    ctx: typer.Context,
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory."),
    run_name: str = typer.Option(..., "--name", help="Run name (runs/{name})."),
    model_size: str = typer.Option("small", help="small | medium | large."),
    config: Path | None = typer.Option(None, help="Pretrain YAML (configs/pretrain.yaml)."),
    runs_root: Path = typer.Option(Path("runs"), help="Runs root directory."),
    resume: str | None = typer.Option(None, help="'auto' to resume runs/{name}/checkpoints/last.pt."),
    wandb: bool = typer.Option(False, "--wandb",
                               help="Mirror metrics to Weights & Biases (needs the [extras] extra)."),
) -> None:
    """Pretrain a pragmatiq model (MLM) on tokenized shards."""
    from pragmatiq import api

    # Only forward non-default flags, so `wandb: true` / `verbose:` in the
    # YAML still win over CLI defaults.
    overrides: dict = {}
    if wandb:
        overrides["wandb"] = True
    if ctx.obj and not ctx.obj.get("verbose", True):
        overrides["verbose"] = False  # --quiet also silences the heartbeat
    summary = api.pretrain(shard_dir, run_name, model_size=model_size, config=config,
                           runs_root=runs_root, resume=resume, **overrides)
    typer.echo(json.dumps(summary, indent=2))


@app.command("probe")
def probe_cmd(
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory."),
    run: Path = typer.Option(..., help="Run directory of a trained model."),
    label: Path = typer.Option(..., help="Label parquet (labels/<task>.parquet)."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
    probe_model: str = typer.Option("gbdt", help="Probe head: gbdt | logistic | lightgbm."),
) -> None:
    """Probe a trained model on a label table; reports ROC-AUC + PR-AUC vs baseline."""
    from pragmatiq import api

    typer.echo(json.dumps(
        api.probe(shard_dir, run, label, device=device, probe_model=probe_model), indent=2))


@app.command("uplift")
def uplift_cmd(
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory."),
    run: Path = typer.Option(..., help="Run directory of a trained model."),
    label: Path = typer.Option(..., help="comm_uplift label parquet (labels/comm_uplift.parquet)."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
    learner: str = typer.Option("t", help="uplift meta-learner: t (two-model) | s (single-model)."),
) -> None:
    """Evaluate campaign uplift on a trained model; reports Qini vs an oracle Qini."""
    from pragmatiq import api

    typer.echo(json.dumps(api.uplift(shard_dir, run, label, device=device, learner=learner), indent=2))


@app.command("finetune")
def finetune_cmd(
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory."),
    run: Path = typer.Option(..., help="Run directory of a trained model."),
    label: Path = typer.Option(..., help="Label parquet (labels/<task>.parquet)."),
    config: Path | None = typer.Option(None, help="Finetune YAML (configs/finetune/*.yaml)."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
) -> None:
    """LoRA fine-tune a trained model's adapters + head on a label table."""
    from pragmatiq import api

    typer.echo(json.dumps(api.finetune(shard_dir, run, label, config=config, device=device), indent=2))


@app.command("embed")
def embed_cmd(
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory."),
    run: Path = typer.Option(..., help="Run directory of a trained model."),
    out: Path = typer.Option(Path("embeddings.parquet"), help="Output parquet."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
) -> None:
    """Embed all users to a parquet of user_id + embedding."""
    from pragmatiq import api

    typer.echo(json.dumps(api.embed(shard_dir, run, out=out, device=device), indent=2))


@app.command("quickstart")
def quickstart_cmd(
    out: Path = typer.Option(Path("runs/quickstart"), help="Output directory."),
    n_users: int = typer.Option(50000, help="Synthetic users."),
    model_size: str = typer.Option("nano", help="Model size for the smoke run."),
    max_steps: int = typer.Option(400, help="Pretrain steps."),
    n_workers: int = typer.Option(0, help="Generation workers."),
) -> None:
    """End-to-end smoke: synth -> tokenize -> pretrain -> probe (prints AUC)."""
    from pragmatiq import api

    res = api.quickstart(out=out, n_users=n_users, model_size=model_size,
                         max_steps=max_steps, n_workers=n_workers)
    typer.echo(json.dumps(res, indent=2))
    typer.echo(res["message"])


@app.command("validate")
def validate_cmd(
    data_dir: Path = typer.Argument(..., help="Raw dataset directory."),
) -> None:
    """Validate a raw dataset against the data contract."""
    from pragmatiq import api

    res = api.validate(data_dir)
    typer.echo(res["summary"])
    raise typer.Exit(code=0 if res["ok"] else 1)


@app.command("export")
def export_cmd(
    run: Path = typer.Option(..., help="Run directory of a trained model."),
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory (for an example user)."),
    out: Path = typer.Option(Path("pragmatiq_embedder.onnx"), help="Output ONNX path."),
) -> None:
    """Export the padded-embedder ONNX variant (varlen caveat documented)."""
    from pragmatiq import api

    typer.echo(json.dumps(api.export(run, shard_dir, out=out), indent=2))


@app.command("benchmark")
def benchmark_cmd(
    run: Path = typer.Option(..., help="Run directory of a trained model."),
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
    out: Path = typer.Option(Path("deploy/benchmarks/RESULTS.md"), help="Results markdown."),
) -> None:
    """Benchmark batch-embedding throughput; writes RESULTS.md."""
    from pragmatiq import api

    typer.echo(json.dumps(api.benchmark(run, shard_dir, device=device, out=out), indent=2))


@app.command("gnn")
def gnn_cmd(
    shard_dir: Path = typer.Argument(..., help="Tokenized shard directory."),
    run: Path = typer.Option(..., help="Run directory of a trained model."),
    transfers: Path = typer.Option(..., help="transfers.parquet."),
    aml_label: Path = typer.Option(..., help="labels/aml.parquet."),
    seeds: str = typer.Option("0,1,2", help="Comma-separated seeds."),
    epochs: int = typer.Option(150, help="GraphSAGE epochs."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
) -> None:
    """Run the four-arm AML GNN ablation (isolated vs GNN+PRAGMA vs GNN+handcrafted vs no-graph control)."""
    from pragmatiq import api

    seed_tuple = tuple(int(s) for s in seeds.split(","))
    res = api.gnn(shard_dir, run, transfers, aml_label, seeds=seed_tuple, epochs=epochs, device=device)
    typer.echo(json.dumps(res, indent=2))


runs_app = typer.Typer(help="Inspect runs.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list(runs_root: Path = typer.Option(Path("runs"), help="Runs root.")) -> None:
    """List runs with their last logged step/loss."""
    from pragmatiq import api

    typer.echo(json.dumps(api.runs_list(runs_root), indent=2))


@runs_app.command("compare")
def runs_compare(
    names: list[str] = typer.Argument(..., help="Run names to compare."),
    runs_root: Path = typer.Option(Path("runs"), help="Runs root."),
) -> None:
    """Compare the last metrics of several runs side by side."""
    from pragmatiq import api

    typer.echo(json.dumps(api.runs_compare(names, runs_root), indent=2))


@synth_app.command("calibrate")
def synth_calibrate(
    stats: Path = typer.Option(..., help="aggregates.yaml with bank-shareable statistics."),
    config: Path | None = typer.Option(None, help="Base WorldConfig YAML to start from."),
    out: Path | None = typer.Option(None, help="Where to write the calibrated config YAML."),
) -> None:
    """Fit generator priors to aggregate statistics (moment matching)."""
    from pragmatiq import api

    result = api.calibrate(stats=stats, config=config, out=out)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
