

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union
from torch import Tensor
from timm.models.swin_transformer_v2 import SwinTransformerV2
from timm.models.layers import trunc_normal_

from src.model.visiomel_model import VisiomelModel
from utils.utils import build_model


logger = logging.getLogger(__name__)


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1
    
    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.
    
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * (patch_size ** 2)
    
    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)
    
    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5
    
    return targets_


class SwinTransformerV2ForSimMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, in_chans, patch_size, norm_target_patch_size=None):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.norm_target_patch_size = norm_target_patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        
        # norm target as prompted
        if self.norm_target_patch_size is not None:
            x = norm_targets(x, self.norm_target_patch_size)
        
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


class SwinTransformerV2SimMIM(VisiomelModel):
    def __init__(
        self, 
        model_name: str, 
        img_size = 224, 
        patch_size: int = 4,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        grad_checkpointing: bool = False,
    ):
        super().__init__(
            optimizer_init=optimizer_init, 
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning, 
            log_norm_verbose=log_norm_verbose,
            lr_layer_decay=lr_layer_decay,
        )
        self.save_hyperparameters()

        if pretrained and patch_size != 4:
            logger.warning(
                f'You are using pretrained model with patch_size={patch_size}. '
                'Pretrained model is trained with patch_size=4. '
                'This will result in resetting of lower layers of the model. '
            )

        encoder = build_model(
            model_name, 
            mock_class=SwinTransformerV2ForSimMIM,
            num_classes=0, 
            img_size=img_size, 
            patch_size=patch_size,
            pretrained=pretrained,
        )
        encoder.set_grad_checkpointing(grad_checkpointing)
        self.model = SimMIM(
            encoder=encoder, 
            encoder_stride=32, 
            in_chans=3, 
            patch_size=patch_size
        )

        # TODO: called in each VisiomelModel subclass but after subclass __init__
        # need to move to VisiomelModel somehow
        self.unfreeze_only_selected()

    def configure_metrics(self):
        """Configure task-specific metrics."""
        return

    def update_train_metrics(self, preds, batch):
        """Update train metrics."""
        return

    def update_val_metrics(self, preds, batch, dataloader_idx=0):
        """Update val metrics."""
        return

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        img, mask, _ = batch
        loss = self.model(img, mask)
        return loss, {'simmim': loss}, None
