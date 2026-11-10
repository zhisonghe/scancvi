import torch
import torchmetrics.functional as tmf

from scvi.module.base import BaseModuleClass, LossOutput
from scvi.train._constants import METRIC_KEYS
from scvi.train._trainingplans import TrainingPlan
from scvi.train._metrics import ElboMetric


class ScancTrainingPlan(TrainingPlan):
    """Lightning module task for SCANCVI Training.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    n_classes
        The number of classes in the labeled dataset.
    classification_ratio
        Weight of the classification_loss in loss function
    lr
        Learning rate used for optimization :class:`~torch.optim.Adam`.
    weight_decay
        Weight decay used in :class:`~torch.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    cache_graph_laplacian
        If True, moves adjacency edge lists to the training device at fit start.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        n_classes: int,
        *,
        classification_ratio: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: str = "elbo_validation",
        compile: bool = False,
        compile_kwargs: dict | None = None,
        cache_graph_laplacian: bool = False,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            compile=compile,
            compile_kwargs=compile_kwargs,
            **loss_kwargs,
        )
        self.loss_kwargs.update({"classification_ratio": classification_ratio})
        self.n_classes = n_classes
        self.cache_graph_laplacian = cache_graph_laplacian

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.cache_graph_laplacian and hasattr(self.module, "cache_graph_laplacian"):
            self.module.cache_graph_laplacian(self.device)

    def log_with_mode(self, key: str, value, mode: str, **kwargs):
        """Log with mode."""
        self.log(f"{mode}_{key}", value, **kwargs)

    def compute_and_log_metrics(
        self, loss_output: LossOutput, metrics: dict[str, ElboMetric], mode: str
    ):
        """Computes and logs metrics."""
        super().compute_and_log_metrics(loss_output, metrics, mode)

        # no labeled observations in minibatch
        if loss_output.classification_loss is None:
            return

        classification_loss = loss_output.classification_loss
        true_labels = loss_output.true_labels.squeeze(-1)
        logits = loss_output.logits
        predicted_labels = torch.argmax(logits, dim=-1)
        # scvi-tools >= 1.4 exposes `on_step`/`on_epoch` plan options; fall back to the
        # previous (and upstream default) epoch-level logging on older versions.
        on_step = getattr(self, "on_step", False)
        on_epoch = getattr(self, "on_epoch", True)

        accuracy = tmf.classification.multiclass_accuracy(
            predicted_labels,
            true_labels,
            self.n_classes,
            average="micro",
        )
        f1 = tmf.classification.multiclass_f1_score(
            predicted_labels,
            true_labels,
            self.n_classes,
            average="micro",
        )
        ce = tmf.classification.multiclass_calibration_error(
            logits,
            true_labels,
            self.n_classes,
        )

        self.log_with_mode(
            METRIC_KEYS.CLASSIFICATION_LOSS_KEY,
            classification_loss,
            mode,
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.ACCURACY_KEY,
            accuracy,
            mode,
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.F1_SCORE_KEY,
            f1,
            mode,
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.CALIBRATION_ERROR_KEY,
            ce,
            mode,
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=loss_output.n_obs_minibatch,
        )

    def training_step(self, batch, batch_idx):
        """Training step for SCANCVI training."""
        if len(batch) == 2:
            full_dataset = batch[0]
            labelled_dataset = batch[1]
        else:
            full_dataset = batch
            labelled_dataset = None

        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        input_kwargs = {
            "labelled_tensors": labelled_dataset,
        }
        input_kwargs.update(self.loss_kwargs)
        _, _, loss_output = self.forward(full_dataset, loss_kwargs=input_kwargs)
        loss = loss_output.loss
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for SCANCVI training."""
        if len(batch) == 2:
            full_dataset = batch[0]
            labelled_dataset = batch[1]
        else:
            full_dataset = batch
            labelled_dataset = None

        input_kwargs = {
            "labelled_tensors": labelled_dataset,
        }
        input_kwargs.update(self.loss_kwargs)
        _, _, loss_output = self.forward(full_dataset, loss_kwargs=input_kwargs)
        loss = loss_output.loss
        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.compute_and_log_metrics(loss_output, self.val_metrics, "validation")

        return loss