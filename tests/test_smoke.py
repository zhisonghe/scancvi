from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scancvi import SCANCVI, SCANCVAE, construct_batch_adjacency_matrices


def make_tiny_adata() -> AnnData:
    counts = np.array(
        [
            [2, 1, 0, 3, 1],
            [1, 0, 4, 2, 2],
            [3, 2, 1, 0, 1],
            [0, 1, 3, 2, 4],
            [4, 2, 0, 1, 1],
            [2, 3, 1, 1, 0],
            [1, 1, 2, 3, 2],
            [0, 2, 4, 1, 3],
            [3, 0, 2, 2, 1],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(["b0", "b0", "b0", "b0", "b0", "b1", "b1", "b1", "b1"]),
            "labels": pd.Categorical([
                "t0",
                "t1",
                "unknown",
                "t0",
                "unknown",
                "t1",
                "t0",
                "t1",
                "unknown",
            ]),
        },
        index=[f"cell_{i}" for i in range(counts.shape[0])],
    )
    adata = AnnData(counts, obs=obs)
    adata.obsm["X_test"] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
            [1.0, 1.0, 0.0],
            [1.1, 1.0, 0.0],
            [1.0, 1.1, 0.0],
            [1.1, 1.1, 0.0],
            [1.2, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return adata


def test_package_import_smoke() -> None:
    module = importlib.import_module("scancvi")

    assert hasattr(module, "__version__")
    assert module.SCANCVI is SCANCVI


def test_model_initialization_smoke() -> None:
    adata = make_tiny_adata()

    SCANCVI.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
        unlabeled_category="unknown",
    )
    model = SCANCVI(adata, n_hidden=16, n_latent=4, n_layers=1, dropout_rate=0.0)

    assert isinstance(model.module, SCANCVAE)
    assert model.summary_stats.n_batch == 2
    assert model.summary_stats.n_labels == 3
    assert model.summary_stats.n_vars == adata.n_vars


def test_construct_batch_adjacency_matrices_smoke() -> None:
    pytest.importorskip("scanpy")
    adata = make_tiny_adata()

    batch_matrices = construct_batch_adjacency_matrices(
        adata,
        batch_key="batch",
        embedding_key="X_test",
        n_neighbors=2,
        show_progress=False,
    )

    assert list(batch_matrices.keys()) == ["b0", "b1"]
    assert batch_matrices["b0"].shape == (5, 5)
    assert batch_matrices["b1"].shape == (4, 4)
    for batch_id, adjacency in batch_matrices.items():
        assert adjacency.nnz > 0, batch_id
        assert adjacency.format == "csr", batch_id