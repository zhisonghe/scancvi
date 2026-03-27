import logging

import numpy as np
import scipy.sparse
from anndata import AnnData

from scvi.utils import track


logger = logging.getLogger(__name__)


def construct_batch_adjacency_matrices(
    adata: AnnData,
    batch_key: str,
    embedding_key: str | None = None,
    n_neighbors: int = 15,
    use_diffmap: bool = False,
    mutual_nn: bool = False,
    recompute_hvg_pca: bool = False,
    layer_hvg: str | None = None,
    n_hvg: int = 2000,
    n_pcs: int = 30,
    diffmap_key: str = "X_diffmap",
    pca_key: str = "X_pca",
    hvg_flavor: str = "seurat_v3",
    show_progress: bool = True,
) -> dict[object, scipy.sparse.spmatrix]:
    """Construct per-batch weighted adjacency matrices for SCANCVI.

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Key in `adata.obs` for batch labels.
    embedding_key
        Key in `adata.obsm` for the embedding used to build the kNN graph. Required
        when `recompute_hvg_pca=False`.
    n_neighbors
        Number of neighbors for the kNN graph.
    use_diffmap
        If True, compute diffusion map and rebuild the kNN graph using `diffmap_key`.
    mutual_nn
        If True, keep only mutual nearest neighbors in the adjacency graph.
    recompute_hvg_pca
        If True, recompute highly variable genes and PCA per batch and use `pca_key`.
    layer_hvg
        Layer to use for highly variable gene selection when `recompute_hvg_pca=True`.
    n_hvg
        Number of highly variable genes to select when `recompute_hvg_pca=True`.
    n_pcs
        Number of PCs (and diffusion components) to compute.
    pca_key
        Key used to store PCA results in `adata.obsm` when recomputing PCA.
    diffmap_key
        Key used to store diffusion map results in `adata.obsm`.
    hvg_flavor
        Flavor argument for `scanpy.pp.highly_variable_genes`.
    show_progress
        Whether to display a progress bar over batches.

    Returns
    -------
    Dict mapping batch IDs to sparse adjacency matrices, one per batch.
    """
    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key} not found in adata.obs")
    if not recompute_hvg_pca and embedding_key is None:
        raise ValueError("embedding_key must be provided when recompute_hvg_pca=False")

    try:
        import scanpy as sc
    except ImportError as err:
        raise ImportError(
            "scanpy is required for construct_batch_adjacency_matrices. "
            "Install it with scvi-tools[scanpy] or scvi-tools[optional]."
        ) from err

    batch_labels = np.asarray(adata.obs[batch_key])
    ordered_batches = list(dict.fromkeys(batch_labels))
    batch_matrices: dict[object, scipy.sparse.spmatrix] = {}

    for batch in track(
        ordered_batches,
        description="Constructing batch adjacency matrices",
        disable=not show_progress,
    ):
        idx = np.where(batch_labels == batch)[0]
        sub = adata[idx].copy()

        if recompute_hvg_pca:
            sc.pp.highly_variable_genes(sub, n_top_genes=n_hvg, flavor=hvg_flavor, layer=layer_hvg)
            sc.pp.pca(sub, n_comps=n_pcs)
            rep_key = pca_key
        else:
            if embedding_key not in sub.obsm:
                raise KeyError(f"{embedding_key} not found in adata.obsm")
            rep_key = embedding_key

        sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep=rep_key)
        if use_diffmap:
            sc.tl.diffmap(sub, n_comps=n_pcs)
            sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep=diffmap_key)

        adj = sub.obsp["connectivities"]
        if mutual_nn:
            if scipy.sparse.issparse(adj):
                adj = adj.tocsr()
                adj = adj.multiply(adj.T > 0)
            else:
                adj = adj * (adj.T > 0)

        if scipy.sparse.issparse(adj):
            adj = adj.tocsr()
        else:
            adj = scipy.sparse.csr_matrix(adj)
        batch_matrices[batch] = adj

    return batch_matrices


def load_saved_scancvi_files(dir_path: str, adata: AnnData, prefix: str | None = None):
    """Load saved SCANCVI files.

    Parameters
    ----------
    dir_path : str
        Directory path.
    adata : AnnData
        Annotated data matrix.
    prefix : str, optional
        Prefix for files.

    Returns
    -------
    Loaded data.
    """
    # Placeholder
    return {}


# Backward-compatible private aliases.
_construct_batch_adjacency_matrices = construct_batch_adjacency_matrices
_load_saved_scancvi_files = load_saved_scancvi_files


__all__ = [
    "construct_batch_adjacency_matrices",
    "load_saved_scancvi_files",
]