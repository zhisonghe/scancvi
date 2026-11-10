# scancvi

Standalone Python package for SCANCVI, an adaptation of SCANVI that regularizes the latent space by penalizing distances between pre-defined within-batch neighboring cells.

## Scope

This package ships the `scancvi` module as an installable distribution for graph-regularized semi-supervised single-cell modeling. SCANCVI extends SCANVI with a within-batch neighbor graph, encouraging cells linked in a pre-defined graph to remain close in the learned latent representation. The package still depends on `scvi-tools` at runtime because the implementation uses `scvi-tools` base classes, training plans, data managers, and internal utilities.

## Requirements

- Python 3.10 to 3.13
- Runtime dependency: `scvi-tools>=1.3.2,<1.6`
- Optional requirement for graph construction in `construct_batch_adjacency_matrices()`: `scanpy>=1.10`

### Python version and scvi-tools version

scvi-tools 1.4 and later require Python 3.12 or newer. SCANCVI itself still supports Python 3.10 and 3.11, but on those versions the installer can only pick scvi-tools 1.3.x (1.3.2 or newer). To use the newest scvi-tools, use Python 3.12 or 3.13.

The upper bound (`<1.6`) is deliberate: SCANCVI builds on internal scvi-tools APIs that can change between minor releases. The bound is raised once a new scvi-tools release has been tested.

### Pinned version in `uv.lock`

`uv.lock` pins `scvi-tools==1.3.2`, the oldest supported release, because it installs on every supported Python version. A `uv sync` environment therefore uses scvi-tools 1.3.2, while `pip install .` picks the newest release allowed by `pyproject.toml` for your Python version. To use a newer scvi-tools in a uv-managed environment:

```bash
uv pip install -U 'scvi-tools>=1.3.2,<1.6'
```

## Installation

Install from the project directory with `pip`:

```bash
pip install .
```

This installs `scvi-tools` automatically as a package dependency.

If you want to use `construct_batch_adjacency_matrices()` to generate within-batch neighboring cell pairs for each batch, install the optional `scanpy` dependency:

```bash
pip install '.[scanpy]'
```

For editable installation during development:

```bash
pip install -e .
pip install -e '.[scanpy]'
```

## Usage

```python
from scancvi import SCANCVI, construct_batch_adjacency_matrices
```

The public API is exported from `src/scancvi/__init__.py`.

## Examples

### Train a Reference SCANCVI Model

This workflow starts from a pretrained `scvi.model.SCVI` reference model, constructs within-batch neighbor graphs, initializes `SCANCVI` from the SCVI weights, and learns a graph-regularized latent space.

`graph_laplacian_weight` controls the strength of the graph regularization term during training: larger values enforce stronger agreement between the latent positions of graph-linked cells, while `0` disables that penalty.

```python
import scvi
import scancvi

adjmat_batches = scancvi.construct_batch_adjacency_matrices(
	adata=adata,
	batch_key="batch",
	recompute_hvg_pca=True,
	layer_hvg="counts",
	hvg_flavor="seurat_v3",
	n_hvg=3000,
	use_diffmap=True,
	show_progress=True,
)

vae = scvi.model.SCVI.load(dir_path="scvi_model", adata=adata)

lvae = scancvi.SCANCVI.from_scvi_model(
	adata=adata,
	graph_adj_matrices=adjmat_batches,
	labels_key="celltype",
	unlabeled_category="Unknown",
	scvi_model=vae,
)

lvae.train(
	max_epochs=20,
	batch_size=2048,
	n_samples_per_label=500,
	early_stopping=True,
	graph_laplacian_weight=10,
)

lvae.save(dir_path="scancvi_model", overwrite=True)
adata.obsm["X_scancvi"] = lvae.get_latent_representation(adata, give_mean=True)
```

### Project Query Data

This workflow loads a trained reference SCANCVI model, maps query cells with `load_query_data()`, and extracts the query latent representation.

```python
import scancvi

vae = scancvi.SCANCVI.load(dir_path="scancvi_model", adata=adata)

vae_q = scancvi.SCANCVI.load_query_data(
	adata=adata_query,
	reference_model=vae,
)

vae_q.train(
	max_epochs=100,
	plan_kwargs={"weight_decay": 0.0},
	check_val_every_n_epoch=10,
)

vae_q.save(dir_path="scancvi_model_scarches", overwrite=True)
adata_query.obsm["X_scancvi"] = vae_q.get_latent_representation(
	adata_query,
	give_mean=True,
)
```

## Notes

- The package is standalone in distribution terms, not in algorithmic dependency terms.
- The test suite passes with `scvi-tools==1.3.2` on Python 3.10 and with `scvi-tools==1.5.2` on Python 3.12 and 3.13. The workflows were also checked with `scvi-tools==1.4.3` on Python 3.12.
- Runtime validation failed with older `scvi-tools` releases tested (`1.2.0`, `1.2.1`, `1.2.2.post2`, `1.3.0`, and `1.3.1`) due to missing or newly eager dependencies and missing internal APIs.
- A `graph_laplacian_weight` passed to `from_scvi_model()` or `load_query_data()` is used by later `train()` calls unless `train()` is given its own `graph_laplacian_weight`.
- `load_query_data(transfer_batch=False)` requires scvi-tools 1.4 or newer; the default (`True`) works with all supported versions.
- SCANCVI counts the unlabeled category as a class, while SCANVI does not. As a result, `to_scanvi_model()` copies only the shared VAE weights (encoder and decoder); the SCANVI classifier and label-conditioned layers keep their initial values. `to_scanvi_model()` also requires at least one labeled category.

## Developer Notes

If you use `uv` for environment management, the equivalent editable install commands are:

```bash
uv pip install -e .
uv pip install -e '.[tests]'
```

Run the tests with:

```bash
python -m pytest
```

GitHub Actions (`.github/workflows/tests.yml`) runs the tests with `scvi-tools==1.3.2` on Python 3.10 and with the newest allowed scvi-tools on Python 3.12 and 3.13.
