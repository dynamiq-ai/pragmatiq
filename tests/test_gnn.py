"""AML GNN tests: transfer-graph build, handcrafted features, GraphSAGE, ablation plumbing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.models.gnn import (
    AmlGNN,
    TransferGraphBuilder,
    _train_val_test_mask,
    handcrafted_node_features,
    run_aml_ablation,
)


class TestSplitDiscipline:
    """The ablation's fairness rests on a stratified train/val/test split that is
    disjoint and complete, so model selection (val) never touches the test set."""

    def test_masks_disjoint_complete_stratified(self) -> None:
        y = torch.tensor([0, 1] * 60 + [0] * 180)  # 300 nodes, 60 positives
        tr, va, te = _train_val_test_mask(len(y), y, seed=0)
        assert int((tr & va).sum()) == int((tr & te).sum()) == int((va & te).sum()) == 0
        assert int((tr | va | te).sum()) == len(y)  # every node assigned exactly once
        for name, m in (("train", tr), ("val", va), ("test", te)):
            assert int(y[m].sum()) > 0, f"{name} split has no positives (not stratified)"

requires_pyg = pytest.mark.skipif(
    importlib.util.find_spec("torch_geometric") is None,
    reason="torch-geometric is required for the AML GNN model tests",
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("p6")
    generate(
        WorldConfig(n_users=600, months=16, n_merchants=1500, mule_ring_count=6, seed=9,
                    eval_month_credit=4, eval_month_short=10),
        out, n_workers=0, write_report=False,
    )
    return out


def _fake_embeddings(user_ids: list[str], dim: int = 16, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {u: rng.standard_normal(dim).astype(np.float32) for u in user_ids}


class TestGraphBuilder:
    def test_build_graph(self, dataset: Path) -> None:
        import pyarrow.parquet as pq

        aml = pq.read_table(dataset / "labels" / "aml.parquet").to_pandas()
        labels = dict(zip(aml["user_id"], aml["label"]))
        emb = _fake_embeddings(list(labels))
        graph = TransferGraphBuilder(dataset / "transfers.parquet").build(emb, labels)
        assert graph.num_nodes == len(emb)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.max() < graph.num_nodes  # edges reference valid nodes
        assert graph.y.sum() >= 1  # some mules present
        assert graph.x.shape == (graph.num_nodes, 16)

    def test_undirected_doubles_edges(self, dataset: Path) -> None:
        import pyarrow.parquet as pq

        aml = pq.read_table(dataset / "labels" / "aml.parquet").to_pandas()
        labels = dict(zip(aml["user_id"], aml["label"]))
        emb = _fake_embeddings(list(labels))
        b = TransferGraphBuilder(dataset / "transfers.parquet")
        d = b.build(emb, labels, undirected=False)
        u = b.build(emb, labels, undirected=True)
        assert u.edge_index.shape[1] == 2 * d.edge_index.shape[1]


class TestHandcrafted:
    def test_feature_shape_and_finite(self, dataset: Path) -> None:
        import pyarrow.parquet as pq

        aml = pq.read_table(dataset / "labels" / "aml.parquet").to_pandas()
        uids = aml["user_id"].tolist()
        feats = handcrafted_node_features(dataset / "transfers.parquet", uids)
        assert feats.shape[0] == len(uids)
        # generic structural features: in_deg, out_deg, total_deg, log_total_amt, mean_amt
        assert feats.shape[1] == 5
        assert np.isfinite(feats).all()

    def test_degree_is_not_a_mule_oracle(self, dataset: Path) -> None:
        import pyarrow.parquet as pq
        from sklearn.metrics import roc_auc_score

        aml = pq.read_table(dataset / "labels" / "aml.parquet").to_pandas()
        uids = aml["user_id"].tolist()
        y = aml["label"].to_numpy()
        feats = handcrafted_node_features(dataset / "transfers.parquet", uids)
        # Mules are degree- and volume-matched to ordinary accounts: the ring's
        # fan-in/layering/cash-out legs draw amounts from the same lognormal as
        # organic P2P, senders are few, and the discriminative structure is the
        # multi-hop layering chain — not any node's 1-hop degree. So no single
        # hand-crafted degree/volume feature should separate mules well; if one
        # did, degree would be a near-oracle and (c) would trivially dominate (b),
        # collapsing the relational-recovery claim into a feature leak.
        in_deg, out_deg = feats[:, 0], feats[:, 1]
        if y.sum() >= 5:
            for name, f in (("in_deg", in_deg), ("out_deg", out_deg)):
                auc = max(roc_auc_score(y, f), roc_auc_score(y, -f))
                assert auc < 0.72, f"degree feature {name} too predictive (AUC {auc:.3f}): ring degree leaks"


@requires_pyg
class TestAmlGNN:
    def test_forward_shapes(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(20, 16)
        edge_index = torch.randint(0, 20, (2, 60))
        model = AmlGNN(16, hidden=32, n_layers=2)
        out = model(x, edge_index)
        assert out.shape == (20, 2)

    def test_three_layers(self) -> None:
        model = AmlGNN(8, hidden=16, n_layers=3)
        out = model(torch.randn(10, 8), torch.randint(0, 10, (2, 30)))
        assert out.shape == (10, 2)

    def test_rejects_bad_depth(self) -> None:
        with pytest.raises(ValueError):
            AmlGNN(8, n_layers=1)


class TestAblationPlumbing:
    @requires_pyg
    def test_ablation_returns_all_setups(self, dataset: Path) -> None:
        import pyarrow.parquet as pq

        aml = pq.read_table(dataset / "labels" / "aml.parquet").to_pandas()
        labels = dict(zip(aml["user_id"], aml["label"]))
        emb = _fake_embeddings(list(labels), dim=24)
        res = run_aml_ablation(dataset / "transfers.parquet", emb, labels, seeds=(0,), epochs=20)
        assert set(res["per_setup"]) == {"a_isolated", "b_gnn_pragma", "c_gnn_handcrafted",
                                         "d_lr_handcrafted"}
        assert "verdict" in res and "pass" in res["verdict"]
        assert "message_passing_adds" in res["verdict"]  # no-graph control arm
        for setup in res["per_setup"].values():
            assert 0.0 <= setup["mean"] <= 1.0

    def test_results_markdown_and_writeback(self, dataset: Path, tmp_path: Path) -> None:
        from pragmatiq.models.gnn import aml_results_markdown, write_aml_report

        res = {
            "per_setup": {
                "a_isolated": {"mean": 0.70, "std": 0.02},
                "b_gnn_pragma": {"mean": 0.85, "std": 0.03},
                "c_gnn_handcrafted": {"mean": 0.78, "std": 0.04},
                "d_lr_handcrafted": {"mean": 0.74, "std": 0.03},
            },
            "raw": {"a_isolated": [0.70, 0.71, 0.69]},
            "verdict": {"b_beats_a": True, "c_beats_a": True, "b_beats_c": True,
                        "paper_ordering": True, "message_passing_adds": True,
                        "graph_recovers_signal": True, "pragma_competitive": True, "pass": True},
            "n_nodes": 3000, "n_mules": 40, "n_edges": 1200, "seeds": [0, 1, 2], "epochs": 150,
        }
        md = aml_results_markdown(res)
        assert "0.850" in md and "ROC-AUC" in md and "(b) > (c) = True" in md
        assert "provenance: n_nodes=3000" in md  # tables are provenance-stamped
        readme = tmp_path / "README.md"
        readme.write_text("# x\n\n## Results\n\n<!-- AML_ABLATION_RESULTS -->\n\nold\n\n## Next\n")
        write_aml_report(res, readme_path=readme, notebook_path=None)
        out = readme.read_text()
        assert "0.850" in out and "## Next" in out and "old" not in out

        notebook = tmp_path / "04_aml_gnn.ipynb"
        notebook.write_text(json.dumps({
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["# Notebook\n"]},
                {"cell_type": "markdown", "metadata": {"tags": ["aml-results"]},
                 "source": ["## Latest ablation result\n\nold\n"]},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }))
        write_aml_report(res, readme_path=None, notebook_path=notebook)
        cells = json.loads(notebook.read_text())["cells"]
        tagged = [c for c in cells if "aml-results" in c.get("metadata", {}).get("tags", [])]
        assert len(tagged) == 1
        assert "old" not in json.dumps(cells)
        assert "0.850" in "".join(tagged[0]["source"])

    def test_writeback_refuses_scale_downgrade(self, tmp_path: Path) -> None:
        from pragmatiq.models.gnn import write_aml_report

        big = ("# x\n\n## Results\n\n<!-- AML_ABLATION_RESULTS -->\n\nfull-scale table\n\n"
               "<sub>provenance: n_nodes=12000, n_edges=9, n_mules=9, seeds=[0], epochs=1, "
               "commit=abc</sub>\n\n## Next\n")
        readme = tmp_path / "README.md"
        readme.write_text(big)
        small = {
            "per_setup": {"a_isolated": {"mean": 0.7, "std": 0.0},
                          "b_gnn_pragma": {"mean": 0.7, "std": 0.0},
                          "c_gnn_handcrafted": {"mean": 0.7, "std": 0.0}},
            "raw": {"a_isolated": [0.7]},
            "verdict": {"b_beats_a": False, "c_beats_a": False, "b_beats_c": False, "pass": False},
            "n_nodes": 3000, "n_mules": 4, "n_edges": 12, "seeds": [0], "epochs": 1,
        }
        write_aml_report(small, readme_path=readme, notebook_path=None)
        assert "full-scale table" in readme.read_text(), "CI-scale must not clobber full-scale"
