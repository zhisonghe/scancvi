# scancvi

Standalone Python package for SCANCVI, an adaptation of SCANVI that regularizes the latent space by penalizing distances between pre-defined within-batch neighboring cells.

## Scope

This package ships the `scancvi` module as an installable distribution for graph-regularized semi-supervised single-cell modeling. SCANCVI extends SCANVI with a within-batch neighbor graph, encouraging cells linked in a pre-defined graph to remain close in the learned latent representation. The package still depends on `scvi-tools` at runtime because the implementation uses `scvi-tools` base classes, training plans, data managers, and internal utilities.

## Requirements

- Python 3.10 to 3.13
- Runtime dependency: `scvi-tools>=1.3.2,<2`
- Optional requirement for graph construction in `construct_batch_adjacency_matrices()`: `scanpy>=1.10`

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

## Notes

- The package is standalone in distribution terms, not in algorithmic dependency terms.
- Runtime validation passed on Python 3.10 and Python 3.13.
- Runtime validation passed with `scvi-tools==1.3.2`.
- Runtime validation failed with older `scvi-tools` releases tested (`1.2.0`, `1.2.1`, `1.2.2.post2`, `1.3.0`, and `1.3.1`) due to missing or newly eager dependencies and missing internal APIs.

## Developer Notes

If you use `uv` for environment management, the equivalent editable install commands are:

```bash
uv pip install -e .
uv pip install -e '.[tests]'
```
