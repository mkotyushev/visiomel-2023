from copy import deepcopy
import logging
from matplotlib import pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
from finetuning_scheduler import FinetuningScheduler
from typing import Any, Dict, List, Optional, Union
from torch import Tensor
import torch
from torch.nn import ModuleDict, ModuleList, CrossEntropyLoss
from pytorch_lightning.cli import instantiate_class
from torchmetrics import CatMetric, Metric
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryStatScores
from pytorch_lightning.utilities import grad_norm

from utils.utils import state_norm, LogLossScore, PenalizedBinaryFBetaScore


logger = logging.getLogger(__name__)


class VisiomelModel(LightningModule):
    def __init__(
        self, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        n_bootstrap: int = 1000,
        skip_nan: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metrics = None
        self.cat_metrics = None

        self.configure_metrics()

    def compute_loss_preds(self, batch, *args, **kwargs):
        """Compute losses and predictions."""

    def configure_metrics(self):
        """Configure task-specific metrics."""

    def bootstrap_metric(self, probas, targets, metric: Metric):
        """Calculate metric on bootstrap samples."""

    @staticmethod
    def check_batch_dims(batch):
        assert all(map(lambda x: len(x) == len(batch[0]), batch)), \
            f'All entities in batch must have the same length, got ' \
            f'{list(map(len, batch))}'

    def remove_nans(self, y, y_pred):
        nan_mask = torch.isnan(y_pred)
        
        if nan_mask.ndim > 1:
            nan_mask = nan_mask.any(dim=1)
        
        if nan_mask.any():
            if not self.hparams.skip_nan:
                raise ValueError(
                    f'Got {nan_mask.sum()} / {nan_mask.shape[0]} nan values in update_metrics. '
                    f'Use skip_nan=True to skip them.'
                )
            logger.warning(
                f'Got {nan_mask.sum()} / {nan_mask.shape[0]} nan values in update_metrics. '
                f'Dropping them & corresponding targets.'
            )
            y_pred = y_pred[~nan_mask]
            y = y[~nan_mask]
        return y, y_pred

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch.
        Could be overriden for custom batch / prediction structure.
        """
        y, y_pred = batch[1].detach(), preds[:, 1].detach().float()
        y, y_pred = self.remove_nans(y, y_pred)
        y_pred = torch.softmax(y_pred, dim=1)
        return y, y_pred

    def update_metrics(self, span, preds, batch):
        """Update train metrics."""
        y, y_proba = self.extract_targets_and_probas_for_metric(preds, batch)
        self.cat_metrics[span]['probas'].update(y_proba)
        self.cat_metrics[span]['targets'].update(y)

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""
        # Unfreeze all layers if freeze period is over
        if self.hparams.finetuning is not None:
            # TODO change to >= somehow
            if self.current_epoch == self.hparams.finetuning['unfreeze_before_epoch']:
                self.unfreeze()

    def on_train_start(self) -> None:
        # Change dataloader num_workers to 10
        # after cache is filled
        self.trainer.datamodule.hparams.num_workers = self.trainer.datamodule.hparams.num_workers_saturated

    def unfreeze_only_selected(self):
        """
        Unfreeze only layers selected by 
        model.finetuning.unfreeze_layer_names_*.
        """
        if self.hparams.finetuning is not None:
            for name, param in self.named_parameters():
                selected = False

                if 'unfreeze_layer_names_startswith' in self.hparams.finetuning:
                    selected = selected or any(
                        name.startswith(pattern) 
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_startswith']
                    )

                if 'unfreeze_layer_names_contains' in self.hparams.finetuning:
                    selected = selected or any(
                        pattern in name
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_contains']
                    )
                logger.info(f'Param {name}\'s requires_grad == {selected}.')
                param.requires_grad = selected

    def training_step(self, batch, batch_idx, **kwargs):
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'train_loss_{loss_name}', 
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch[0].shape[0],
            )
        self.update_metrics('train_metrics', preds, batch)

        # Handle nan in loss
        has_nan = False
        if torch.isnan(total_loss):
            has_nan = True
            logger.warning(
                f'Loss is nan at epoch {self.current_epoch} '
                f'step {self.global_step}.'
            )
        for loss_name, loss in losses.items():
            if torch.isnan(loss):
                has_nan = True
                logger.warning(
                    f'Loss {loss_name} is nan at epoch {self.current_epoch} '
                    f'step {self.global_step}.'
                )
        if has_nan:
            return None
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one val dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'val_loss_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=batch[0].shape[0],
            )
        self.update_metrics('val_metrics', preds, batch)
        return total_loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        _, _, preds = self.compute_loss_preds(batch, **kwargs)
        return preds

    def log_metrics_and_reset(
        self, 
        prefix, 
        on_step=False, 
        on_epoch=True, 
        prog_bar_names=None,
        reset=True,
    ):
        # Get metric span: train or val
        span = None
        if prefix == 'train':
            span = 'train_metrics'
        elif prefix in ['val', 'val_ds']:
            span = 'val_metrics'
        
        # Get concatenated preds and targets
        # and reset them
        probas, targets = \
            self.cat_metrics[span]['probas'].compute().cpu(),  \
            self.cat_metrics[span]['targets'].compute().cpu()
        if reset:
            self.cat_metrics[span]['probas'].reset()
            self.cat_metrics[span]['targets'].reset()

        # Calculate and log metrics
        for name, metric in self.metrics.items():
            metric_value = None
            if prefix == 'val_ds':  # bootstrap
                if self.hparams.n_bootstrap > 0:
                    metric_value = self.bootstrap_metric(probas[:, 1], targets, metric)
                else:
                    logger.warning(
                        f'prefix == val_ds but n_bootstrap == 0. '
                        f'No bootstrap metrics will be calculated '
                        f'and logged.'
                    )
            else:
                metric.update(probas[:, 1], targets)
                metric_value = metric.compute()
                metric.reset()
            
            prog_bar = False
            if prog_bar_names is not None:
                prog_bar = (name in prog_bar_names)

            if metric_value is not None:
                if type(metric) == BinaryStatScores:
                    for key, value in zip(['tp', 'fp', 'tn', 'fn', 'sup'], metric_value):
                        self.log(
                            f'{prefix}_{name}_{key}',
                            value,
                            on_step=on_step,
                            on_epoch=on_epoch,
                            prog_bar=prog_bar,
                        )
                else:
                    self.log(
                        f'{prefix}_{name}',
                        metric_value,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=prog_bar,
                    )

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
        if self.metrics is None:
            return
        assert self.cat_metrics is not None

        prog_bar_names = ['f1', 'ce']
        self.log_metrics_and_reset(
            'train',
            on_step=False,
            on_epoch=True,
            prog_bar_names=prog_bar_names,
            reset=True,
        )
    
    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        if self.metrics is None:
            return
        assert self.cat_metrics is not None

        prog_bar_names = ['f1']
        self.log_metrics_and_reset(
            'val',
            on_step=False,
            on_epoch=True,
            prog_bar_names=prog_bar_names,
            reset=False,
        )
        self.log_metrics_and_reset(
            'val_ds',
            on_step=False,
            on_epoch=True,
            prog_bar_names=prog_bar_names,
            reset=True,
        )

    def get_lr_decayed(self, lr, layer_index, layer_name):
        """
        Get lr decayed by 
            - layer index as (self.hparams.lr_layer_decay ** layer_index) if
              self.hparams.lr_layer_decay is float 
              (useful e. g. when new parameters are in classifer head)
            - layer name as self.hparams.lr_layer_decay[layer_name] if
              self.hparams.lr_layer_decay is dict
              (useful e. g. when pretrained parameters are at few start layers 
              and new parameters are the most part of the model)
        """
        if isinstance(self.hparams.lr_layer_decay, dict):
            for key in self.hparams.lr_layer_decay:
                if layer_name.startswith(key):
                    return lr * self.hparams.lr_layer_decay[key]
            return lr
        elif isinstance(self.hparams.lr_layer_decay, float):
            if self.hparams.lr_layer_decay == 1.0:
                return lr
            else:
                return lr * (self.hparams.lr_layer_decay ** layer_index)

    def build_parameter_groups(self):
        """Get parameter groups for optimizer."""
        names, params = list(zip(*self.named_parameters()))
        num_layers = len(params)
        grouped_parameters = [
            {
                'params': param, 
                'lr': self.get_lr_decayed(
                    self.hparams.optimizer_init['init_args']['lr'], 
                    num_layers - layer_index - 1,
                    name
                )
            } for layer_index, (name, param) in enumerate(self.named_parameters())
        ]
        logger.info(
            f'Number of layers: {num_layers}, '
            f'min lr: {names[0]}, {grouped_parameters[0]["lr"]}, '
            f'max lr: {names[-1]}, {grouped_parameters[-1]["lr"]}'
        )
        return grouped_parameters

    def configure_optimizer(self):
        optimizer = instantiate_class(args=self.build_parameter_groups(), init=self.hparams.optimizer_init)
        return optimizer

    def fts_callback(self):
        for c in self.trainer.callbacks:
            if isinstance(c, FinetuningScheduler):
                return c
        return None

    def configure_lr_scheduler(self, optimizer):
        # Convert milestones from total persents to steps
        # for PiecewiceFactorsLRScheduler
        if (
            'PiecewiceFactorsLRScheduler' in self.hparams.lr_scheduler_init['class_path'] and
            self.hparams.pl_lrs_cfg['interval'] == 'step'
        ):
            # max_epochs is number of epochs of current FTS stage
            # if corresponding FTS callback is used
            fts_callback = self.fts_callback()
            if fts_callback is not None and 'max_transition_epoch' in fts_callback.ft_schedule[fts_callback.curr_depth]:
                max_epochs = fts_callback.ft_schedule[self.curr_depth]['max_transition_epoch']
            else:
                max_epochs = self.trainer.max_epochs

            total_steps = len(self.trainer.fit_loop._data_source.dataloader()) * max_epochs
            grad_accum_steps = self.trainer.accumulate_grad_batches
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * total_steps / grad_accum_steps) 
                for milestone in self.hparams.lr_scheduler_init['init_args']['milestones']
            ]
        
        scheduler = instantiate_class(args=optimizer, init=self.hparams.lr_scheduler_init)
        scheduler = {
            "scheduler": scheduler,
            **self.hparams.pl_lrs_cfg,
        }

        return scheduler

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        if self.hparams.lr_scheduler_init is None:
            return optimizer

        scheduler = self.configure_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms."""
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'grad_2.0_norm_total' in norms:
                self.log('grad_2.0_norm_total', norms['grad_2.0_norm_total'])

        norms = state_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'state_2.0_norm_total' in norms:
                self.log('state_2.0_norm_total', norms['state_2.0_norm_total'])


class VisiomelClassifier(VisiomelModel):
    def __init__(
        self, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        label_smoothing: float = 0.0,
        n_bootstrap: int = 1000,
        skip_nan: bool = False,
    ):
        super().__init__(
            optimizer_init=optimizer_init,
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning,
            log_norm_verbose=log_norm_verbose,
            lr_layer_decay=lr_layer_decay,
            n_bootstrap=n_bootstrap,
            skip_nan=skip_nan,
        )
        self.save_hyperparameters()
        self.loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    def bootstrap_metric(self, probas, targets, metric: Metric):
        """Calculate metric on bootstrap samples."""
        assert self.hparams.num_classes == 2, 'Only binary classification is supported.'

        neg_indices = torch.arange(targets.shape[0])[targets == 0]
        pos_indices = torch.arange(targets.shape[0])[targets == 1]
        
        # TODO: try to code vectorized version
        metric_values = []
        for _ in range(self.hparams.n_bootstrap):
            neg_indices_sample = neg_indices[torch.randperm(neg_indices.shape[0])[:pos_indices.shape[0]]]
            indices = torch.cat([neg_indices_sample, pos_indices])
            metric.update(probas[indices], targets[indices])
            metric_values.append(
                metric.compute()
            )
            metric.reset()
        
        if isinstance(metric_values[0], Tensor) and metric_values[0].ndim > 0:  # ndim tensors
            return torch.stack(metric_values, dim=0).float().mean(dim=0)
        elif isinstance(metric_values[0], Tensor) and metric_values[0].ndim == 0:  # scalars tensors
            return torch.tensor(metric_values).float().mean(dim=0)
        else:  # scalar floats / ints / numpy arrays
            return torch.from_numpy(np.array(metric_values)).float().mean(dim=0)

    def configure_metrics(self):
        """Configure task-specific metrics."""
        self.metrics = ModuleDict(
            {
                'll': LogLossScore(),
                'auc': BinaryAUROC(),
                'f1': BinaryF1Score(),
                'bss': BinaryStatScores(),
                'pf1s': PenalizedBinaryFBetaScore(mode='soft', beta=1.0),
            }
        )
        self.cat_metrics = ModuleDict(
            {
                'train_metrics': ModuleDict(
                    {
                        'probas': CatMetric(),
                        'targets': CatMetric()
                    }
                ),
                'val_metrics': ModuleDict(
                    {
                        'probas': CatMetric(),
                        'targets': CatMetric()
                    }
                ),
            }
        )
