import inspect
import logging
import warnings
from copy import deepcopy
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import torch
from anndata import AnnData
from torch.utils.data import DataLoader
from itertools import cycle

import scvi
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import _SETUP_ARGS_KEY, ADATA_MINIFY_TYPE
from scvi.data._utils import _is_minified
from scvi.data.fields import CategoricalObsField, LabelsWithUnlabeledObsField, LayerField, NumericalObsField
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.model._utils import get_max_epochs_heuristic, parse_device_args, use_distributed_sampler
from scvi.model.base import ArchesMixin, BaseModelClass, VAEMixin
from scvi.train import Trainer, TrainRunner
from scvi.train._callbacks import SubSampleLabels

from ._module import SCANCVAE
from ._task import ScancTrainingPlan

logger = logging.getLogger(__name__)

# obs column created by SCANCVI.setup_anndata when no labels_key is given.
_UNLABELED_COLUMN = "_scancvi_labels"

if TYPE_CHECKING:
    from scvi.model._scvi import SCVI
    from scvi.model._scanvi import SCANVI


class SCANCVI(ArchesMixin, BaseModelClass, VAEMixin):
    """
    SCANVI with Cell-to-cell Connectivity (SCANCVI) model.

    Adapts the SCANVI model for semi-supervised learning with cell-to-cell connectivity.

    Parameters
    ----------
    adata : AnnData
        AnnData object that has been registered via :meth:`~scvi.model.SCANCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
    **model_kwargs
        Keyword args for :class:`~scvi.external.scancvi.SCANCVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCANCVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    >>> model = scvi.model.SCANCVI(adata)
    >>> model.train()
    >>> latent = model.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["nb"] = "nb",
        **model_kwargs,
    ) -> None:
        super().__init__(adata)

        # Initialize the SCANVI-like module
        self.module = SCANCVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_labels=self.summary_stats.n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCANCVI Model with the following params: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, dropout_rate: {dropout_rate}"
        )
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: "SCVI",
        unlabeled_category: str,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        graph_adj_matrices: (
            list[np.ndarray | torch.Tensor]
            | dict[object, np.ndarray | torch.Tensor]
            | None
        ) = None,
        **scancvi_kwargs,
    ):
        """Initialize SCANCVI model with weights from pretrained :class:`~scvi.model.SCVI` model.

        Parameters
        ----------
        scvi_model
            Pretrained scvi model.
        labels_key
            Key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the SCVI model. If None, uses the `labels_key` used to
            setup the SCVI model. If that was None, an error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with SCANCVI.
        adata
            AnnData object that has been registered via :meth:`~scvi.external.SCANCVI.setup_anndata`.
        graph_adj_matrices
            Weighted adjacency matrices. If a list, one per batch ordered by batch index.
            If a dict, keys are batch IDs (category values from `batch_key` or integer batch
            indices) and values are adjacency matrices. Each matrix should be square and ordered
            by the cells in that batch as they appear in `adata`.
        scancvi_kwargs
            kwargs for SCANCVI model.
        """
        scvi_model._check_if_trained(message="Passed in scvi model hasn't been trained yet.")

        scancvi_kwargs = dict(scancvi_kwargs)
        graph_laplacian_weight = scancvi_kwargs.pop("graph_laplacian_weight", 0.0)
        init_params = scvi_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in scancvi_kwargs.keys():
                warnings.warn(
                    f"Ignoring param '{k}' as it was already passed in to pretrained "
                    f"SCVI model with value {v}.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                del scancvi_kwargs[k]

        if scvi_model.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise ValueError(
                "We cannot use the given scVI model to initialize SCANCVI because it has "
                "minified adata. Keep counts when minifying model using "
                "minified_data_type='latent_posterior_parameters_with_counts'."
            )

        if adata is None:
            adata = scvi_model.adata
        elif adata:
            if _is_minified(adata):
                raise ValueError("Please provide a non-minified `adata` to initialize SCANCVI.")
            scvi_model._validate_anndata(adata)
        else:
            adata = None

        scvi_setup_args = deepcopy(scvi_model.registry[_SETUP_ARGS_KEY])
        allowed_setup_keys = {
            "batch_key",
            "labels_key",
            "layer",
            "categorical_covariate_keys",
            "continuous_covariate_keys",
        }
        scvi_setup_args = {k: v for k, v in scvi_setup_args.items() if k in allowed_setup_keys}
        scvi_labels_key = scvi_setup_args["labels_key"]
        if labels_key is None and scvi_labels_key is None:
            raise ValueError(
                "A `labels_key` is necessary as the scVI model was initialized without one."
            )
        if scvi_labels_key is None:
            scvi_setup_args.update({"labels_key": labels_key})

        if adata is not None:
            cls.setup_anndata(
                adata,
                unlabeled_category=unlabeled_category,
                **scvi_setup_args,
            )

        scancvi_model = cls(adata, **non_kwargs, **kwargs, **scancvi_kwargs)
        scvi_state_dict = scvi_model.module.state_dict()
        scancvi_model.module.load_state_dict(scvi_state_dict, strict=False)
        scancvi_model.was_pretrained = True
        scancvi_model._graph_laplacian_weight = graph_laplacian_weight
        if graph_adj_matrices is not None:
            scancvi_model.set_graph_laplacian(graph_adj_matrices)

        return scancvi_model

    @classmethod
    def load_query_data(
        cls,
        adata=None,
        reference_model=None,
        registry=None,
        inplace_subset_query_vars: bool = False,
        accelerator: str = "auto",
        device: int | str = "auto",
        unfrozen: bool = False,
        freeze_dropout: bool = False,
        freeze_expression: bool = True,
        freeze_decoder_first_layer: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        freeze_classifier: bool = True,
        transfer_batch: bool = True,
        datamodule=None,
        graph_adj_matrices: (
            list[np.ndarray | torch.Tensor]
            | dict[object, np.ndarray | torch.Tensor]
            | None
        ) = None,
        graph_laplacian_weight: float | None = None,
    ):
        """Load query data with scArches and optionally attach query graph regularization.

        This wraps :meth:`~scvi.model.base.ArchesMixin.load_query_data` and adds SCANCVI-specific
        options for query graph Laplacian regularization.

        Parameters
        ----------
        transfer_batch
            Passed to :meth:`~scvi.model.base.ArchesMixin.load_query_data`. Setting it to
            ``False`` requires scvi-tools >= 1.4; the default (``True``) works with all
            supported versions.
        graph_adj_matrices
            Optional weighted adjacency matrices for the query data. If provided, they are
            attached to the returned SCANCVI model via :meth:`set_graph_laplacian`.
        graph_laplacian_weight
            Optional default graph Laplacian loss weight to store on the returned model for
            subsequent :meth:`train` calls.
        """
        extra_kwargs = {}
        if not transfer_batch:
            if "transfer_batch" not in inspect.signature(ArchesMixin.load_query_data).parameters:
                raise ValueError(
                    "`transfer_batch=False` requires scvi-tools >= 1.4 "
                    f"(installed: {scvi.__version__})."
                )
            extra_kwargs["transfer_batch"] = transfer_batch

        model = super().load_query_data(
            adata=adata,
            reference_model=reference_model,
            registry=registry,
            inplace_subset_query_vars=inplace_subset_query_vars,
            accelerator=accelerator,
            device=device,
            unfrozen=unfrozen,
            freeze_dropout=freeze_dropout,
            freeze_expression=freeze_expression,
            freeze_decoder_first_layer=freeze_decoder_first_layer,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_classifier=freeze_classifier,
            datamodule=datamodule,
            **extra_kwargs,
        )

        if graph_adj_matrices is not None:
            model.set_graph_laplacian(graph_adj_matrices)
        if graph_laplacian_weight is not None:
            model._graph_laplacian_weight = graph_laplacian_weight

        return model

    def to_scvi_model(self) -> "SCVI":
        """Convert SCANCVI model to a SCVI model with the same architecture and weights.

        This method creates a new SCVI model using the same data (adata) and compatible
        architecture parameters as the SCANCVI model, then loads the core VAE weights
        (encoder and decoder) from the SCANCVI module.

        Returns
        -------
        SCVI model with the same architecture and core weights as this SCANCVI model.

        Raises
        ------
        ValueError
            If the SCANCVI model has not been trained yet.

        Examples
        --------
        >>> scancvi_model = scvi.external.SCANCVI(adata)
        >>> scancvi_model.train()
        >>> scvi_model = scancvi_model.to_scvi_model()
        >>> latent = scvi_model.get_latent_representation()
        """
        from scvi.model import SCVI

        self._check_if_trained(message="SCANCVI model must be trained before conversion to SCVI.")

        # Get SCANCVI init params
        scancvi_params = self.init_params_
        scancvi_non_kwargs = scancvi_params["non_kwargs"]
        scancvi_kwargs = scancvi_params["kwargs"]
        scancvi_kwargs = {k: v for (i, j) in scancvi_kwargs.items() for (k, v) in j.items()}

        # Extract parameters that are compatible with SCVI
        scvi_init_params = {
            "n_hidden": scancvi_non_kwargs.get("n_hidden", 128),
            "n_latent": scancvi_non_kwargs.get("n_latent", 10),
            "n_layers": scancvi_non_kwargs.get("n_layers", 1),
            "dropout_rate": scancvi_non_kwargs.get("dropout_rate", 0.1),
            "dispersion": scancvi_non_kwargs.get("dispersion", "gene"),
            "gene_likelihood": scancvi_non_kwargs.get("gene_likelihood", "nb"),
        }

        # Setup the SCVI model with the same AnnData
        SCVI.setup_anndata(
            self.adata,
            batch_key=self.adata_manager.registry[_SETUP_ARGS_KEY].get("batch_key"),
            layer=self.adata_manager.registry[_SETUP_ARGS_KEY].get("layer"),
            categorical_covariate_keys=self.adata_manager.registry[_SETUP_ARGS_KEY].get(
                "categorical_covariate_keys"
            ),
            continuous_covariate_keys=self.adata_manager.registry[_SETUP_ARGS_KEY].get(
                "continuous_covariate_keys"
            ),
        )

        # Create the SCVI model
        scvi_model = SCVI(self.adata, **scvi_init_params)

        # Load compatible weights from SCANCVI module into SCVI module
        # SCANCVI.module is SCANCVAE which inherits from VAE, so core VAE weights are compatible
        scvi_state_dict = scvi_model.module.state_dict()
        scancvi_state_dict = self.module.state_dict()

        # Load only weights present in SCVI.
        # This keeps shared VAE components and skips semisupervised heads that SCVI does not
        # have (e.g., classifier, encoder_z2_z1, decoder_z1_z2, y_prior, ...).
        compatible_weights = {}
        for key, value in scancvi_state_dict.items():
            if key in scvi_state_dict and scvi_state_dict[key].shape == value.shape:
                compatible_weights[key] = value

        scvi_model.module.load_state_dict(compatible_weights, strict=False)
        # Match the inference behavior of a trained model (batch norm uses running stats).
        scvi_model.module.eval()
        scvi_model.is_trained_ = True
        scvi_model.trainer = self.trainer

        return scvi_model

    def to_scanvi_model(self) -> "SCANVI":
        """Convert SCANCVI model to a SCANVI model with the same architecture and weights.

        This method creates a new SCANVI model using the same data (adata) and compatible
        architecture parameters as the SCANCVI model, then loads the core semi-supervised VAE
        weights (encoder and decoder) from the SCANCVI module. SCANCVI-specific components
        (graph Laplacian regularization) are not transferred.

        Returns
        -------
        SCANVI model with the same semi-supervised architecture and core weights as this
        SCANCVI model.

        Raises
        ------
        ValueError
            If the SCANCVI model has not been trained yet.

        Examples
        --------
        >>> scancvi_model = scvi.external.SCANCVI(adata)
        >>> scancvi_model.train()
        >>> scanvi_model = scancvi_model.to_scanvi_model()
        >>> latent = scanvi_model.get_latent_representation()
        >>> predictions = scanvi_model.predict()
        """
        from scvi.model import SCANVI

        self._check_if_trained(message="SCANCVI model must be trained before conversion to SCANVI.")

        # Get SCANCVI init params
        scancvi_params = self.init_params_
        scancvi_non_kwargs = scancvi_params["non_kwargs"]
        scancvi_kwargs = scancvi_params["kwargs"]
        scancvi_kwargs = {k: v for (i, j) in scancvi_kwargs.items() for (k, v) in j.items()}

        # Get the unlabeled category from SCANCVI setup
        unlabeled_category = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).unlabeled_category
        labels_key = self.adata_manager.registry[_SETUP_ARGS_KEY].get("labels_key")
        if labels_key is None or self.summary_stats.n_labels < 2:
            raise ValueError(
                "Cannot convert to SCANVI: the model has no labeled categories "
                "(SCANVI needs at least one label besides the unlabeled category)."
            )

        # Extract parameters that are compatible with SCANVI
        scanvi_init_params = {
            "n_hidden": scancvi_non_kwargs.get("n_hidden", 128),
            "n_latent": scancvi_non_kwargs.get("n_latent", 10),
            "n_layers": scancvi_non_kwargs.get("n_layers", 1),
            "dropout_rate": scancvi_non_kwargs.get("dropout_rate", 0.1),
            "dispersion": scancvi_non_kwargs.get("dispersion", "gene"),
            "gene_likelihood": scancvi_non_kwargs.get("gene_likelihood", "nb"),
        }

        # Setup the SCANVI model with the same AnnData
        SCANVI.setup_anndata(
            self.adata,
            labels_key=labels_key,
            unlabeled_category=unlabeled_category,
            batch_key=self.adata_manager.registry[_SETUP_ARGS_KEY].get("batch_key"),
            layer=self.adata_manager.registry[_SETUP_ARGS_KEY].get("layer"),
            categorical_covariate_keys=self.adata_manager.registry[_SETUP_ARGS_KEY].get(
                "categorical_covariate_keys"
            ),
            continuous_covariate_keys=self.adata_manager.registry[_SETUP_ARGS_KEY].get(
                "continuous_covariate_keys"
            ),
        )

        # Create the SCANVI model
        # The unlabeled category is recorded by ``setup_anndata`` above; SCANVI.__init__ would
        # forward an ``unlabeled_category`` kwarg to the module, which rejects it.
        scanvi_model = SCANVI(self.adata, **scanvi_init_params)

        # Load compatible weights from SCANCVI module into SCANVI module
        # Both SCANCVI and SCANVI use semi-supervised VAE, so their core weights are compatible
        scanvi_state_dict = scanvi_model.module.state_dict()
        scancvi_state_dict = self.module.state_dict()

        # Load all matching SCANVI-compatible weights.
        # In practice this includes semisupervised heads as well
        # (classifier, encoder_z2_z1, decoder_z1_z2, y_prior, ...).
        # Graph Laplacian structures are runtime attributes in SCANCVI and are not part of the
        # state dict, so they are not transferred.
        compatible_weights = {}
        for key, value in scancvi_state_dict.items():
            if key in scanvi_state_dict and scanvi_state_dict[key].shape == value.shape:
                compatible_weights[key] = value

        scanvi_model.module.load_state_dict(compatible_weights, strict=False)
        # Match the inference behavior of a trained model (batch norm uses running stats).
        scanvi_model.module.eval()
        scanvi_model.is_trained_ = True
        scanvi_model.trainer = self.trainer

        return scanvi_model

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        labels_key: str | None = None,
        unlabeled_category: str = "unknown",
        layer: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ) -> None:
        """
        Set up AnnData for SCANCVI.

        Parameters
        ----------
        adata
            AnnData object containing raw counts.
        batch_key
            Key in `adata.obs` for batch information.
        labels_key
            Key in `adata.obs` for label information.
        unlabeled_category
            Category for unlabeled cells in `labels_key`.
        layer
            Layer in `adata.layers` to use for raw counts.
        categorical_covariate_keys
            Keys in `adata.obs` for categorical covariates.
        continuous_covariate_keys
            Keys in `adata.obs` for continuous covariates.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        # Always register labels with an unlabeled category, since training uses
        # SemiSupervisedDataSplitter, which requires it. Without a labels_key, every cell is
        # treated as unlabeled via a generated column (re-created whenever setup is replayed).
        if labels_key is None:
            labels_key = _UNLABELED_COLUMN
            adata.obs[labels_key] = pd.Categorical([unlabeled_category] * adata.n_obs)
        label_field = LabelsWithUnlabeledObsField(
            REGISTRY_KEYS.LABELS_KEY,
            labels_key,
            unlabeled_category,
        )
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            label_field,
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if categorical_covariate_keys is not None:
            anndata_fields.extend(
                [
                    CategoricalObsField(REGISTRY_KEYS.CAT_COVS_KEY, cov_key)
                    for cov_key in categorical_covariate_keys
                ]
            )
        if continuous_covariate_keys is not None:
            anndata_fields.extend(
                [
                    NumericalObsField(REGISTRY_KEYS.CONT_COVS_KEY, cov_key)
                    for cov_key in continuous_covariate_keys
                ]
            )
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int | None = None,
        use_gpu: int | bool | str | None = None,
        n_samples_per_label: int | None = None,
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        graph_laplacian_weight: float | None = None,
        cache_graph_laplacian: bool = False,
        **trainer_kwargs,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of epochs to train.
        use_gpu
            Use GPU for training.
        n_samples_per_label
            Number of labeled examples to subsample per class in each training epoch.
        train_size
            Size of training set.
        validation_size
            Size of validation set.
        batch_size
            Batch size for training.
        early_stopping
            Whether to use early stopping.
        datasplitter_kwargs
            Additional keyword arguments passed into
            :class:`~scvi.dataloaders.SemiSupervisedDataSplitter`.
        plan_kwargs
            Keyword arguments for training plan.
        graph_laplacian_weight
            Weight for the graph Laplacian regularization term. Set to 0 to disable. If
            ``None``, uses the weight stored by :meth:`from_scvi_model` or
            :meth:`load_query_data` (``graph_laplacian_weight=...``), or 0 if none was stored.
        cache_graph_laplacian
            If True, moves adjacency edge lists to the training device at fit start.
        trainer_kwargs
            Keyword arguments for trainer.
        """
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        if use_gpu is None:
            accelerator = "auto"
            devices = "auto"
        elif isinstance(use_gpu, bool):
            accelerator = "auto" if use_gpu else "cpu"
            devices = "auto"
        elif isinstance(use_gpu, int):
            accelerator = "gpu"
            devices = [use_gpu]
        elif isinstance(use_gpu, str):
            if use_gpu in {"auto", "cpu", "gpu", "cuda", "mps", "tpu"}:
                accelerator = use_gpu
                devices = "auto"
            elif use_gpu.startswith("cuda:"):
                accelerator = "cuda"
                devices = [int(use_gpu.split(":", maxsplit=1)[1])]
            elif use_gpu.isdigit():
                accelerator = "gpu"
                devices = [int(use_gpu)]
            else:
                accelerator = "gpu"
                devices = use_gpu
        else:
            accelerator = "auto"
            devices = "auto"

        accelerator, devices, _ = parse_device_args(
            accelerator=accelerator,
            devices=devices,
            return_device="torch",
        )

        plan_kwargs = plan_kwargs or {}
        datasplitter_kwargs = datasplitter_kwargs or {}
        plan_kwargs.setdefault("n_classes", self.summary_stats.n_labels)
        if graph_laplacian_weight is None:
            graph_laplacian_weight = getattr(self, "_graph_laplacian_weight", 0.0)
        plan_kwargs.setdefault("graph_laplacian_weight", graph_laplacian_weight)
        plan_kwargs.setdefault("cache_graph_laplacian", cache_graph_laplacian)

        data_splitter = SemiSupervisedDataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            n_samples_per_label=n_samples_per_label,
            distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = ScancTrainingPlan(self.module, **plan_kwargs)

        callbacks = trainer_kwargs.pop("callbacks", [])
        if len(data_splitter._labeled_indices) != 0:
            callbacks.append(SubSampleLabels())
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        trainer_kwargs["enable_progress_bar"] = True

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

    def _validate_anndata(self, adata: AnnData | None = None, copy_if_view: bool = True):
        """Add SCANCVI's generated obs columns to a new AnnData before transferring setup.

        ``setup_anndata`` creates ``_indices`` (and, without ``labels_key``, an all-unlabeled
        label column). scvi-tools transfers the setup to a new AnnData without re-running
        ``setup_anndata``, so these columns must exist beforehand.
        """
        if adata is not None and self.get_anndata_manager(adata) is None:
            setup_args = self.registry_[_SETUP_ARGS_KEY]
            if "_indices" not in adata.obs:
                adata.obs["_indices"] = np.arange(adata.n_obs)
            if setup_args.get("labels_key") is None and _UNLABELED_COLUMN not in adata.obs:
                adata.obs[_UNLABELED_COLUMN] = pd.Categorical(
                    [setup_args.get("unlabeled_category", "unknown")] * adata.n_obs
                )
        return super()._validate_anndata(adata, copy_if_view=copy_if_view)

    def set_graph_laplacian(
        self,
        graph_adj_matrices: (
            list[np.ndarray | torch.Tensor]
            | dict[object, np.ndarray | torch.Tensor]
            | None
        ),
    ) -> None:
        """
        Attach per-batch adjacency matrices for graph Laplacian regularization.

        If a dict is provided, keys are batch IDs (category values from `batch_key` or
        integer batch indices) and values are adjacency matrices.
        """
        if graph_adj_matrices is None:
            self.module.set_graph_laplacian(None, None)
            return

        n_batch = max(1, int(self.summary_stats.n_batch))
        if isinstance(graph_adj_matrices, dict):
            batch_mapping = self.adata_manager.get_state_registry(
                REGISTRY_KEYS.BATCH_KEY
            ).categorical_mapping
            ordered: list[np.ndarray | torch.Tensor] = []
            for i, batch_id in enumerate(batch_mapping):
                if batch_id in graph_adj_matrices:
                    ordered.append(graph_adj_matrices[batch_id])
                elif i in graph_adj_matrices:
                    ordered.append(graph_adj_matrices[i])
                else:
                    ordered.append(None)
            missing_keys = [
                batch_id
                for batch_id, adj in zip(batch_mapping, ordered, strict=True)
                if adj is None
            ]
            if missing_keys:
                warnings.warn(
                    "graph_adj_matrices is missing entries for batch IDs: "
                    f"{missing_keys}. Filling with empty adjacency matrices.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            graph_adj_matrices = ordered

        if len(graph_adj_matrices) != n_batch:
            raise ValueError(
                "Length of graph_adj_matrices must equal number of batches "
                f"({n_batch}), but got {len(graph_adj_matrices)}."
            )

        batch_indices = (
            self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).astype(int).ravel()
        )
        index_maps: list[np.ndarray] = []
        adj_matrices: list[np.ndarray | torch.Tensor | sp_sparse.spmatrix] = []

        for b in range(n_batch):
            batch_cells = np.where(batch_indices == b)[0]
            adj = graph_adj_matrices[b]
            if adj is None:
                adj = sp_sparse.csr_matrix(
                    (batch_cells.shape[0], batch_cells.shape[0]), dtype=float
                )
            adj_matrices.append(adj)
            if adj.shape[0] != adj.shape[1]:
                raise ValueError(f"Adjacency matrix for batch {b} must be square.")
            if adj.shape[0] != batch_cells.shape[0]:
                raise ValueError(
                    "Adjacency matrix size does not match number of cells in batch "
                    f"{b}: expected {batch_cells.shape[0]}, got {adj.shape[0]}."
                )
            index_map = -np.ones(self.adata.n_obs, dtype=int)
            index_map[batch_cells] = np.arange(batch_cells.shape[0])
            index_maps.append(index_map)

        self.module.set_graph_laplacian(adj_matrices, index_maps)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: int | None = None,
        return_dist: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Get latent representation.

        Parameters
        ----------
        adata
            AnnData object to get latent for.
        indices
            Indices of cells to get latent for.
        give_mean
            If ``True``, returns the mean of the latent distribution. If ``False``, returns a
            sampled latent representation.
        mc_samples
            Number of Monte Carlo samples to estimate the mean for distributions with no
            closed-form mean (e.g., logistic normal). Used when ``give_mean`` is ``True`` and
            ``return_dist`` is ``False``.
        batch_size
            Batch size for inference.
        return_dist
            If ``True``, returns the mean and variance of the latent distribution.

        Returns
        -------
        Latent representation, or a tuple of latent mean and variance when
        ``return_dist=True``.
        """
        from torch.distributions import Normal
        from torch.nn.functional import softmax

        from scvi.module._constants import MODULE_KEYS

        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latent = []
        qz_means = []
        qz_vars = []
        for tensors in dataloader:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if MODULE_KEYS.QZ_KEY in outputs:
                qz = outputs.get(MODULE_KEYS.QZ_KEY)
                qzm = qz.loc
                qzv = qz.scale.square()
            else:
                qzm = outputs.get(MODULE_KEYS.QZM_KEY)
                qzv = outputs.get(MODULE_KEYS.QZV_KEY)
                qz = Normal(qzm, qzv.sqrt())

            if return_dist:
                qz_means.append(qzm.cpu().numpy())
                qz_vars.append(qzv.cpu().numpy())
                continue

            z = qzm if give_mean else outputs.get(MODULE_KEYS.Z_KEY)

            if give_mean and getattr(self.module, "latent_distribution", None) == "ln":
                samples = qz.sample([mc_samples])
                z = softmax(samples, dim=-1).mean(dim=0)

            latent.append(z.cpu().numpy())

        if return_dist:
            return np.concatenate(qz_means, axis=0), np.concatenate(qz_vars, axis=0)
        return np.concatenate(latent, axis=0)

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Get normalized expression.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Indices of cells.
        n_samples_overall
            Number of samples.
        batch_size
            Batch size.
        return_mean
            Whether to return mean.

        Returns
        -------
        Normalized expression.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        exprs = []
        for tensors in dataloader:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
            outputs = self.module.generative(**generative_inputs)
            px = outputs["px"]
            # px.scale: normalized expression (sums to 1 per cell);
            # px.mu: expected counts (scale multiplied by library size).
            expr = px.scale if return_mean else px.mu
            exprs.append(expr.cpu().numpy())
        return np.concatenate(exprs, axis=0)


class TrainDL(DataLoader):
    """Train data loader."""

    def __init__(self, data_loader_list, **kwargs):
        self.data_loader_list = data_loader_list
        self.largest_train_dl_idx = np.argmax([len(dl.indices) for dl in data_loader_list])
        self.largest_dl = self.data_loader_list[self.largest_train_dl_idx]
        super().__init__(self.largest_dl, **kwargs)

    def __len__(self):
        return len(self.largest_dl)

    def __iter__(self):
        train_dls = [
            dl if i == self.largest_train_dl_idx else cycle(dl)
            for i, dl in enumerate(self.data_loader_list)
        ]
        return zip(*train_dls, strict=True)