import logging
from finetuning_scheduler import FinetuningScheduler
from typing import Any, Dict, List, Optional
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleDict
from pytorch_lightning.cli import instantiate_class
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from pytorch_lightning.utilities import grad_norm

from utils.utils import state_norm, build_classifier, CrossEntropyScore


logger = logging.getLogger(__name__)


class SwinTransformerV2Classifier(LightningModule):
    def __init__(
        self, 
        model_name: str, 
        num_classes: int = 2, 
        img_size = 224, 
        patch_size: int = 4,
        patch_embed_backbone_name: Optional[str] = None,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if pretrained and patch_size != 4:
            logger.warning(
                f'You are using pretrained model with patch_size={patch_size}. '
                'Pretrained model is trained with patch_size=4. '
                'This will result in resetting of lower layers of the model. '
            )
        
        self.model = build_classifier(
            model_name, 
            num_classes, 
            patch_embed_backbone_name=patch_embed_backbone_name, 
            img_size=img_size, 
            patch_size=patch_size,
            pretrained=pretrained
        )
        self.unfreeze_only_selected()

        self.loss_fn = CrossEntropyLoss()
        self.metrics = ModuleDict(
            {
                'accuracy': BinaryAccuracy(),
                'f1': BinaryF1Score(),
                'cross_entropy': CrossEntropyScore()
            }
        )

    def unfreeze_only_selected(self):
        if self.hparams.finetuning is not None:
            for name, param in self.model.named_parameters():
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

    def on_train_epoch_end(self) -> None:
        if self.hparams.finetuning is not None:
            if self.current_epoch >= self.hparams.finetuning['unfreeze_after_epoch']:
                self.unfreeze()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def compute_loss_preds(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, preds

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, _ = self.compute_loss_preds(batch)
        self.log(
            'train_loss', 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, preds = self.compute_loss_preds(batch)
        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        y, y_pred = batch[1], preds[:, 1]
        for _, metric in self.metrics.items():
            metric.update(y_pred, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        for name, metric in self.metrics.items():
            prog_bar = (name == 'cross_entropy')
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
            for _, p in self.model.named_parameters()
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