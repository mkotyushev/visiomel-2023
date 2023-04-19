import logging
import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
from finetuning_scheduler import FinetuningScheduler
from typing import Any, Dict, List, Optional
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleDict
from mock import patch
from timm.models.swin_transformer_v2 import SwinTransformerV2
from pytorch_lightning.cli import instantiate_class
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
from pytorch_lightning.utilities import grad_norm

from utils.utils import load_pretrained, state_norm
from model.patch_embed_with_backbone import PatchEmbedWithBackbone


logger = logging.getLogger(__name__)


class SwinTransformerV2WithBackbone(SwinTransformerV2):
    r""" Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
            - https://arxiv.org/abs/2111.09883
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(
            self, patch_embed_backbone: nn.Module,
            img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            window_size=7, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            pretrained_window_sizes=(0, 0, 0, 0), **kwargs):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            pretrained_window_sizes,
            **kwargs
        )

        self.patch_embed = PatchEmbedWithBackbone(
            backbone=patch_embed_backbone,
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


def build_classifier(
    model_name, 
    num_classes, 
    img_size, 
    patch_size, 
    patch_embed_backbone_name=None,
    pretrained=True
):   
    # Load pretrained model with its default img_size
    # and then load the pretrained weights to the model via
    # load_pretrained function by Swin V2 authors
    pretrained_model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )
    if patch_embed_backbone_name is not None:
        patch_embed_backbone = timm.create_model(
            patch_embed_backbone_name, pretrained=pretrained, num_classes=0
        )
        with patch('timm.models.swin_transformer_v2.SwinTransformerV2', SwinTransformerV2WithBackbone):
            model = timm.create_model(
                model_name, 
                pretrained=False, 
                num_classes=num_classes, 
                img_size=img_size, 
                patch_embed_backbone=patch_embed_backbone, 
                patch_size=patch_size
            )
    else:
        model = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=num_classes, 
            img_size=img_size, 
            patch_size=patch_size
        )
    model = load_pretrained(pretrained_model.state_dict(), model)

    del pretrained_model
    torch.cuda.empty_cache()

    return model


class CrossEntropyScore(Metric):
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[])
        self.add_state("target", default=[])

    def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
        return torch.stack((1 - preds, preds), dim=1), target

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape[0] == target.shape[0]

        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        self.preds = torch.cat(self.preds, dim=0)
        self.target = torch.cat(self.target, dim=0)
        return F.cross_entropy(self.preds, self.target).item()


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