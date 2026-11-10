"""End-to-end workflow tests on a tiny synthetic dataset (CPU, 1-2 epochs)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import scvi
from anndata import AnnData
from scvi.model.base import ArchesMixin

from scancvi import SCANCVI

N_LATENT = 4
# SCVI.train takes `accelerator`; SCANCVI.train takes `use_gpu`.
SCVI_TRAIN_KW = {"accelerator": "cpu", "enable_progress_bar": False, "batch_size": 64}
TRAIN_KW = {"use_gpu": False, "batch_size": 64}


def make_adata(n_obs: int, batches: tuple[str, ...], seed: int) -> AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n_obs, 30)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(rng.choice(list(batches), n_obs)),
            "celltype": pd.Categorical(rng.choice(["A", "B", "C", "Unknown"], n_obs)),
        },
        index=[f"s{seed}_c{i}" for i in range(n_obs)],
    )
    adata = AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(30)]))
    adata.layers["counts"] = adata.X.copy()
    return adata


def knn_graphs(adata: AnnData, k: int = 5, seed: int = 0) -> dict[str, sp.csr_matrix]:
    """Random symmetric kNN-like graph per batch (keyed by batch category)."""
    rng = np.random.default_rng(seed)
    graphs = {}
    for batch in adata.obs["batch"].cat.categories:
        n = int((adata.obs["batch"] == batch).sum())
        rows = np.repeat(np.arange(n), k)
        cols = rng.integers(0, n, size=n * k)
        adj = sp.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
        adj = ((adj + adj.T) > 0).astype(float)
        adj.setdiag(0)
        adj.eliminate_zeros()
        graphs[batch] = adj.tocsr()
    return graphs


@pytest.fixture(scope="module")
def reference():
    scvi.settings.seed = 0
    adata = make_adata(240, ("b0", "b1"), seed=0)
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch", layer="counts")
    vae = scvi.model.SCVI(adata, n_latent=N_LATENT, n_hidden=16)
    vae.train(max_epochs=1, **SCVI_TRAIN_KW)

    model = SCANCVI.from_scvi_model(
        adata=adata,
        scvi_model=vae,
        labels_key="celltype",
        unlabeled_category="Unknown",
        graph_adj_matrices=knn_graphs(adata),
    )
    model.train(
        max_epochs=2,
        n_samples_per_label=10,
        graph_laplacian_weight=10.0,
        check_val_every_n_epoch=1,
        **TRAIN_KW,
    )
    return adata, model


def test_train_logs_positive_graph_loss(reference) -> None:
    _, model = reference
    history = model.history
    for key in ("graph_laplacian_loss_train", "graph_laplacian_loss_validation"):
        assert key in history
        assert float(history[key].iloc[-1, 0]) > 0
    assert "train_classification_loss" in history


def test_train_with_cached_graph(reference) -> None:
    _, model = reference
    model.train(
        max_epochs=1, graph_laplacian_weight=1.0, cache_graph_laplacian=True, **TRAIN_KW
    )
    assert float(model.history["graph_laplacian_loss_train"].iloc[-1, 0]) > 0


def test_latent_and_expression_outputs(reference) -> None:
    adata, model = reference
    z = model.get_latent_representation(adata)
    qzm, qzv = model.get_latent_representation(adata, return_dist=True)
    assert z.shape == qzm.shape == qzv.shape == (adata.n_obs, N_LATENT)
    assert np.all(qzv > 0)

    scale = model.get_normalized_expression(adata)
    assert scale.shape == adata.shape
    np.testing.assert_allclose(scale.sum(axis=1), 1.0, rtol=1e-4)
    mu = model.get_normalized_expression(adata, return_mean=False)
    assert mu.shape == adata.shape
    assert np.all(mu >= 0)


def test_save_load_and_query_with_graph(reference, tmp_path) -> None:
    adata, model = reference
    model.save(tmp_path / "ref", overwrite=True)
    loaded = SCANCVI.load(tmp_path / "ref", adata=adata)

    query = make_adata(120, ("q0",), seed=1)
    query_model = SCANCVI.load_query_data(
        adata=query,
        reference_model=loaded,
        # Only the query batch is provided; reference batches get empty graphs.
        graph_adj_matrices=knn_graphs(query, seed=1),
        graph_laplacian_weight=5.0,
    )
    # No graph_laplacian_weight passed: the weight stored by load_query_data is used.
    query_model.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0}, **TRAIN_KW)
    assert float(query_model.history["graph_laplacian_loss_train"].iloc[-1, 0]) > 0
    assert query_model.get_latent_representation(query).shape == (query.n_obs, N_LATENT)

    # Inference on an AnnData the model was not set up with.
    unseen = make_adata(50, ("b1",), seed=6)
    assert loaded.get_latent_representation(unseen).shape == (unseen.n_obs, N_LATENT)

    # Loading from a path (no graph) must also work.
    query2 = make_adata(60, ("q1",), seed=2)
    SCANCVI.load_query_data(adata=query2, reference_model=str(tmp_path / "ref"))


def test_transfer_batch_flag(reference) -> None:
    _, model = reference
    # With transfer_batch=False, query batches must already exist in the reference.
    query = make_adata(60, ("b0",), seed=3)
    supported = "transfer_batch" in inspect.signature(ArchesMixin.load_query_data).parameters
    if supported:
        SCANCVI.load_query_data(adata=query, reference_model=model, transfer_batch=False)
    else:
        with pytest.raises(ValueError, match="transfer_batch"):
            SCANCVI.load_query_data(adata=query, reference_model=model, transfer_batch=False)


def test_conversions(reference) -> None:
    adata, model = reference
    expected = model.get_latent_representation(adata)
    scvi_model = model.to_scvi_model()
    np.testing.assert_allclose(
        scvi_model.get_latent_representation(), expected, rtol=1e-4, atol=1e-5
    )

    scanvi_model = model.to_scanvi_model()
    predictions = scanvi_model.predict()
    assert len(predictions) == adata.n_obs
    np.testing.assert_allclose(
        scanvi_model.get_latent_representation(), expected, rtol=1e-4, atol=1e-5
    )


def test_from_scratch_without_labels() -> None:
    adata = make_adata(120, ("b0", "b1"), seed=4)
    SCANCVI.setup_anndata(adata, batch_key="batch", layer="counts")
    model = SCANCVI(adata, n_latent=N_LATENT, n_hidden=16)
    model.set_graph_laplacian(knn_graphs(adata, seed=4))
    assert model.summary_stats.n_labels == 1
    model.train(max_epochs=1, graph_laplacian_weight=1.0, **TRAIN_KW)
    assert model.get_latent_representation().shape == (adata.n_obs, N_LATENT)
    with pytest.raises(ValueError, match="no labeled categories"):
        model.to_scanvi_model()

    # The generated label column is re-created when the setup is replayed on new data.
    other = make_adata(60, ("b0",), seed=5)
    assert model.get_latent_representation(other).shape == (other.n_obs, N_LATENT)
