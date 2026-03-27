from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import torch
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F

from scvi import REGISTRY_KEYS
from scvi.module.base import LossOutput, SupervisedModuleClass
from scvi.nn import Decoder, Encoder

from scvi.module._classifier import Classifier
from scvi.module._utils import broadcast_labels
from scvi.module._vae import VAE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from torch.distributions import Distribution


class SCANCVAE(SupervisedModuleClass, VAE):
    """SCANVI with Cell-to-cell Connectivity (SCANCVI).

    This is an adaptation of the scANVI model with cell-to-cell connectivity features.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    use_observed_lib_size
        If ``True``, use the observed library size for RNA as the scaling factor in the mean of the
        conditional distribution.
    y_prior
        If None, initialized to uniform probability over cell types
    labels_groups
        Label group designations
    use_labels_groups
        Whether to use the label groups
    linear_classifier
        If `True`, uses a single linear layer for classification instead of a
        multi-layer perceptron.
    classifier_parameters
        Keyword arguments passed into :class:`~scvi.module.Classifier`.
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    **vae_kwargs
        Keyword args for :class:`~scvi.module.VAE`
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        use_observed_lib_size: bool = True,
        y_prior: torch.Tensor | None = None,
        labels_groups: Sequence[int] = None,
        use_labels_groups: bool = False,
        linear_classifier: bool = False,
        classifier_parameters: dict | None = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        **vae_kwargs,
    ):
        super().__init__(
            n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            n_batch=n_batch,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            use_observed_lib_size=use_observed_lib_size,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **vae_kwargs,
        )

        classifier_parameters = classifier_parameters or {}
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.n_labels = n_labels
        # Classifier takes n_latent as input
        cls_parameters = {
            "n_layers": 0 if linear_classifier else n_layers,
            "n_hidden": 0 if linear_classifier else n_hidden,
            "dropout_rate": dropout_rate,
            "logits": True,
        }
        cls_parameters.update(classifier_parameters)
        self.classifier = Classifier(
            n_latent,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **cls_parameters,
        )

        self.encoder_z2_z1 = Encoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        self.decoder_z1_z2 = Decoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        self.y_prior = torch.nn.Parameter(
            y_prior if y_prior is not None else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )
        self.use_labels_groups = use_labels_groups
        self.labels_groups = np.array(labels_groups) if labels_groups is not None else None
        if self.use_labels_groups:
            if labels_groups is None:
                raise ValueError("Specify label groups")
            unique_groups = np.unique(self.labels_groups)
            self.n_groups = len(unique_groups)
            if not (unique_groups == np.arange(self.n_groups)).all():
                raise ValueError()
            self.classifier_groups = Classifier(
                n_latent, n_hidden, self.n_groups, n_layers, dropout_rate
            )
            self.groups_index = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.tensor(
                            (self.labels_groups == i).astype(np.uint8),
                            dtype=torch.uint8,
                        ),
                        requires_grad=False,
                    )
                    for i in range(self.n_groups)
                ]
            )

        self.graph_adj_matrices: list[tuple[torch.Tensor, torch.Tensor, int]] | None = None
        self.graph_index_maps: list[torch.Tensor] | None = None
        self._graph_adj_device: torch.device | None = None
        self._graph_index_maps_device: torch.device | None = None

    def set_graph_laplacian(
        self,
        graph_adj_matrices: list[torch.Tensor | np.ndarray | scipy.sparse.spmatrix] | None,
        graph_index_maps: list[torch.Tensor] | None,
    ) -> None:
        """Register adjacency matrices and index maps for graph Laplacian regularization."""
        if graph_adj_matrices is None or graph_index_maps is None:
            self.graph_adj_matrices = None
            self.graph_index_maps = None
            self._graph_adj_device = None
            self._graph_index_maps_device = None
            return

        self.graph_adj_matrices = []
        for adj in graph_adj_matrices:
            if isinstance(adj, torch.Tensor):
                if adj.is_sparse:
                    adj = adj.coalesce()
                    edge_index = adj.indices().long()
                    edge_weight = adj.values().float()
                    n_nodes = adj.shape[0]
                else:
                    edge_index = adj.nonzero(as_tuple=False).t().long()
                    edge_weight = adj[edge_index[0], edge_index[1]].float()
                    n_nodes = adj.shape[0]
            elif scipy.sparse.issparse(adj):
                adj = adj.tocoo()
                adj.sum_duplicates()
                edge_index = torch.stack(
                    [torch.from_numpy(adj.row), torch.from_numpy(adj.col)], dim=0
                ).long()
                edge_weight = torch.from_numpy(adj.data).float()
                n_nodes = adj.shape[0]
            else:
                adj = scipy.sparse.coo_matrix(adj)
                adj.sum_duplicates()
                edge_index = torch.stack(
                    [torch.from_numpy(adj.row), torch.from_numpy(adj.col)], dim=0
                ).long()
                edge_weight = torch.from_numpy(adj.data).float()
                n_nodes = adj.shape[0]

            self.graph_adj_matrices.append((edge_index, edge_weight, n_nodes))

        self.graph_index_maps = [
            torch.as_tensor(index_map, dtype=torch.long)
            for index_map in graph_index_maps
        ]
        self._graph_adj_device = None
        self._graph_index_maps_device = None

    def cache_graph_laplacian(self, device: torch.device) -> None:
        """Cache adjacency edge lists on the specified device to avoid per-batch transfers."""
        if self.graph_adj_matrices is None:
            return
        cached: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for edge_index, edge_weight, n_nodes in self.graph_adj_matrices:
            cached.append((edge_index.to(device), edge_weight.to(device), n_nodes))
        self.graph_adj_matrices = cached
        self._graph_adj_device = device
        if self.graph_index_maps is not None:
            self.graph_index_maps = [m.to(device) for m in self.graph_index_maps]
            self._graph_index_maps_device = device

    def _graph_laplacian_loss(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
        cell_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute graph Laplacian loss for a minibatch."""
        if self.graph_adj_matrices is None or self.graph_index_maps is None:
            return torch.zeros([], device=z.device)

        loss = torch.zeros([], device=z.device)
        n_batches = 0
        for b in torch.unique(batch_index):
            batch_id = int(b.item())
            mask = batch_index == b
            if mask.sum() < 2:
                continue
            index_map = self.graph_index_maps[batch_id]
            if index_map.device != z.device:
                index_map = index_map.to(device=z.device)
            local_indices = index_map[cell_indices[mask]]
            valid_mask = local_indices >= 0
            if valid_mask.sum() < 2:
                continue
            local_indices = local_indices[valid_mask]
            edge_index, edge_weight, n_nodes = self.graph_adj_matrices[batch_id]
            if edge_index.device != z.device:
                edge_index = edge_index.to(device=z.device)
                edge_weight = edge_weight.to(device=z.device)

            local_to_mb = torch.full(
                (n_nodes,), -1, dtype=torch.long, device=z.device
            )
            local_to_mb[local_indices] = torch.arange(
                local_indices.shape[0], device=z.device
            )

            src = edge_index[0]
            dst = edge_index[1]
            src_mb = local_to_mb[src]
            dst_mb = local_to_mb[dst]
            edge_mask = (src_mb >= 0) & (dst_mb >= 0)
            if edge_mask.sum() == 0:
                continue

            src_mb = src_mb[edge_mask]
            dst_mb = dst_mb[edge_mask]
            weights = edge_weight[edge_mask]
            weight_sum = weights.sum()
            if weight_sum <= 0:
                continue

            z_b = z[mask][valid_mask]
            diff = z_b[src_mb] - z_b[dst_mb]
            dist2 = (diff * diff).sum(dim=1)
            loss = loss + 0.5 * (weights * dist2).sum() / (weight_sum + 1e-8)
            n_batches += 1

        if n_batches == 0:
            return torch.zeros([], device=z.device)
        return loss / n_batches

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_ouputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
        graph_laplacian_weight: float = 0.0,
    ):
        """Compute the loss."""
        px: Distribution = generative_ouputs["px"]
        qz1: torch.Tensor = inference_outputs["qz"]
        z1: torch.Tensor = inference_outputs["z"]
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        batch_index: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]

        ys, z1s = broadcast_labels(z1, n_broadcast=self.n_labels)
        qz2, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
        reconst_loss = -px.log_prob(x).sum(-1)

        # KL Divergence
        mean = torch.zeros_like(qz2.loc)
        scale = torch.ones_like(qz2.scale)

        kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=-1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)

        probs = self.classifier(z1)
        if self.classifier.logits:
            probs = F.softmax(probs, dim=-1)

        if z1.ndim == 2:
            loss_z1_unweight_ = loss_z1_unweight.view(self.n_labels, -1).t()
            kl_divergence_z2_ = kl_divergence_z2.view(self.n_labels, -1).t()
        else:
            loss_z1_unweight_ = torch.transpose(
                loss_z1_unweight.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
            kl_divergence_z2_ = torch.transpose(
                kl_divergence_z2.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
        reconst_loss += loss_z1_weight + (loss_z1_unweight_ * probs).sum(dim=-1)
        kl_divergence = (kl_divergence_z2_ * probs).sum(dim=-1)
        kl_divergence += kl(
            Categorical(probs=probs),
            Categorical(
                probs=self.y_prior.repeat(probs.size(0), probs.size(1), 1)
                if len(probs.size()) == 3
                else self.y_prior.repeat(probs.size(0), 1)
            ),
        )

        if not self.use_observed_lib_size:
            ql = inference_outputs["ql"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence)

        kl_divergence += kl_divergence_l

        loss = torch.mean(reconst_loss + kl_divergence * kl_weight)

        graph_laplacian_loss = torch.zeros([], device=z1.device)
        if graph_laplacian_weight and graph_laplacian_weight > 0:
            cell_indices = tensors.get(REGISTRY_KEYS.INDICES_KEY)
            if cell_indices is None:
                raise ValueError(
                    "Graph Laplacian regularization requires REGISTRY_KEYS.INDICES_KEY "
                    "in tensors. Ensure setup_anndata adds cell indices."
                )
            if self.graph_adj_matrices is None or self.graph_index_maps is None:
                raise ValueError(
                    "Graph Laplacian regularization requires adjacency matrices to be set "
                    "via SCANCVI.set_graph_laplacian()."
                )
            graph_laplacian_loss = self._graph_laplacian_loss(
                z1,
                batch_index.long().ravel(),
                cell_indices.long().ravel(),
            )
            loss = loss + graph_laplacian_weight * graph_laplacian_loss

        # a payload to be used during autotune
        if self.extra_payload_autotune:
            extra_metrics_payload = {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
        else:
            extra_metrics_payload = {}

        if graph_laplacian_weight and graph_laplacian_weight > 0:
            extra_metrics_payload["graph_laplacian_loss"] = graph_laplacian_loss

        if labelled_tensors is not None:
            ce_loss, true_labels, logits = self.classification_loss(labelled_tensors)

            loss += ce_loss * classification_ratio
            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_loss,
                kl_local=kl_divergence,
                classification_loss=ce_loss,
                true_labels=true_labels,
                logits=logits,
                extra_metrics=extra_metrics_payload,
            )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_divergence,
            extra_metrics=extra_metrics_payload,
        )