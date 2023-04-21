from pytorch_lightning import LightningModule
from finetuning_scheduler import FinetuningScheduler
from typing import Any, Dict, List, Optional
from torch import Tensor
from torch.nn import ModuleDict
from pytorch_lightning.cli import instantiate_class
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from pytorch_lightning.utilities import grad_norm

from utils.utils import state_norm, CrossEntropyScore


class VisiomelModel(LightningModule):
    def __init__(
        self, 
        num_classes: int = 2, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_metrics = ModuleDict(
            {
                'acc': BinaryAccuracy(),
                'f1': BinaryF1Score(),
                'ce': CrossEntropyScore()
            }
        )
        self.val_metrics = ModuleDict(
            {
                'acc': BinaryAccuracy(),
                'f1': BinaryF1Score(),
                'ce': CrossEntropyScore()
            }
        )
        self.val_metrics_downsampled = ModuleDict(
            {
                'ds_acc': BinaryAccuracy(),
                'ds_ce': CrossEntropyScore()
            }
        )

    def compute_loss_preds(self, batch, *args, **kwargs):
        """Compute losses and predictions."""

    def update_train_metrics(self, preds, batch):
        """Update train metrics."""
        y, y_pred = batch[1].detach(), preds[:, 1].detach()
        for _, metric in self.train_metrics.items():
            metric.update(y_pred, y)

    def update_val_metrics(self, preds, batch, dataloader_idx=0):
        """Update val metrics."""
        y, y_pred = batch[1].detach(), preds[:, 1].detach()
        if dataloader_idx == 0:
            for _, metric in self.val_metrics.items():
                metric.update(y_pred, y)
        else:
            for _, metric in self.val_metrics_downsampled.items():
                metric.update(y_pred, y)

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""
        # Unfreeze all layers if freeze period is over
        if self.hparams.finetuning is not None:
            if self.current_epoch >= self.hparams.finetuning['unfreeze_before_epoch']:
                self.unfreeze()
            else:
                self.unfreeze_only_selected()

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
            )
        self.update_train_metrics(preds, batch)
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'val_loss_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.update_val_metrics(preds, batch, dataloader_idx)
        return total_loss

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
        for name, metric in self.train_metrics.items():
            prog_bar = (name == 'f1' or name == 'ce')
            self.log(
                f'train_{name}',
                metric.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
            )
            metric.reset()
    
    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        for name, metric in self.val_metrics.items():
            prog_bar = (name == 'f1')
            self.log(
                f'val_{name}',
                metric.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
            )
            metric.reset()
        for name, metric in self.val_metrics_downsampled.items():
            prog_bar = (name == 'ds_ce')
            self.log(
                f'val_{name}',
                metric.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
            )
            metric.reset()

    def _init_param_groups(self) -> List[Dict]:
        """Initialize the parameter groups. 
        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        return [
            p
            for _, p in self.named_parameters()
            if p.requires_grad
        ]

    def configure_optimizer(self):
        optimizer = instantiate_class(args=self._init_param_groups(), init=self.hparams.optimizer_init)
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
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * total_steps) 
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
            self.log('grad_2.0_norm_total', norms['grad_2.0_norm_total'])

        norms = state_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            self.log('state_2.0_norm_total', norms['state_2.0_norm_total'])