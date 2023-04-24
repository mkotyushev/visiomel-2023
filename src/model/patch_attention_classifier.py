import timm
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union

from model.patch_embed_with_backbone import PatchBackbone
from model.visiomel_model import VisiomelClassifier


class PatchAttentionPooling(nn.Module):
    def __init__(self, n_classes, embed_dim=1536):
        super().__init__()
        self.class_emb = nn.Parameter(torch.randn(1, n_classes, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, dropout=0.2, batch_first=True
        )

    def forward(self, x):
        B = x.shape[0]
        query = self.class_emb.repeat(B, 1, 1)
        x = self.attention(query, x, x, need_weights=False)
        return x


class PatchAttentionClassifier(VisiomelClassifier):
    def __init__(
        self, 
        num_classes: int = 2,
        patch_embed_backbone_name: str = 'swinv2_base_window12to24_192to384_22kft1k',
        patch_size: int = 1536,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
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

        backbone = timm.create_model(patch_embed_backbone_name, pretrained=True, num_classes=0)
        self.patch_embed = PatchBackbone(
            backbone=backbone, 
            patch_size=patch_size, 
            embed_dim=backbone.num_features
        )
        self.pooling = PatchAttentionPooling(
            n_classes=num_classes if num_classes > 2 else 1, 
            embed_dim=backbone.num_features
        )
        self.classifier = nn.Linear(backbone.num_features, 2)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        return loss, {'ce': loss}, out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, L, E)
        x = self.patch_embed(x)
        # (B, L, E) -> (B, C, E)
        x = self.pooling(x)
        # (B, C, E) -> (B, C, 2)
        x = self.classifier(x)
        # (B, C, 2) -> (B, 2) if C == 1 else (B, C, 2) -> (B, C, 2)
        x = x.squeeze(1)
        
        return x
    