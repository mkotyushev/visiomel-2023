import logging
import timm
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, AdamW
from mock import patch
from timm.models.swin_transformer_v2 import SwinTransformerV2

from utils.utils import load_pretrained
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
        pretrained: bool = True
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
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def compute_loss(self, batch):
        x, y = batch
        x = self(x)
        loss = self.loss_fn(x, y)
        return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self.compute_loss(batch)
        self.log(
            'train_loss', 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self.compute_loss(batch)
        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=1e-4)
