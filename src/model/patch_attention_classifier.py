import timm
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union

from model.patch_embed_with_backbone import PatchBackbone
from model.visiomel_model import VisiomelClassifier


class PatchAttentionPooling(nn.Module):
    def __init__(self, n_classes, embed_dim=1536, hidden_dim=64):
        super().__init__()
        self.class_emb = nn.Parameter(torch.randn(1, n_classes, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            kdim=embed_dim,
            vdim=embed_dim,
            num_heads=1, 
            dropout=0.2, 
            batch_first=True
        )

    def forward(self, x):
        B = x.shape[0]
        query = self.class_emb.repeat(B, 1, 1)
        x, _ = self.attention(query, x, x)
        return x


class PatchAttentionClassifier(VisiomelClassifier):
    def __init__(
        self, 
        num_classes: int = 2,
        patch_embed_backbone_name: str = 'swinv2_base_window12to24_192to384_22kft1k',
        patch_embed_backbone_ckpt_path: str = None,
        patch_size: int = 1536,
        patch_batch_size: int = 1,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        grad_checkpointing: bool = False,
        attention_hidden_dim: int = 64,
        patch_embed_caching: bool = False,
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

        backbone = timm.create_model(
            patch_embed_backbone_name, 
            img_size=patch_size, 
            pretrained=False, 
            num_classes=0
        )
        if patch_embed_backbone_ckpt_path is not None:
            # If backbone is fine-tuned then it is done via SwinTransformerV2SimMIM
            # module, so we need to remove the prefix 'model.encoder.' from the
            # checkpoint state_dict keys.
            state_dict = {
                k \
                    .replace('model.encoder.', 'model.'): v 
                for k, v in 
                torch.load(patch_embed_backbone_ckpt_path)['state_dict'].items()
            }
            backbone.load_state_dict(state_dict, strict=False)
        backbone.set_grad_checkpointing(grad_checkpointing)
        
        self.patch_embed = PatchBackbone(
            backbone=backbone, 
            patch_size=patch_size, 
            embed_dim=backbone.num_features,
            patch_batch_size=patch_batch_size,
            patch_embed_caching=patch_embed_caching,
        )
        self.pooling = PatchAttentionPooling(
            n_classes=num_classes if num_classes > 2 else 1, 
            embed_dim=backbone.num_features,
            hidden_dim=attention_hidden_dim
        )
        self.classifier = nn.Linear(attention_hidden_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.patch_embed_caching = patch_embed_caching

        self.unfreeze_only_selected()
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        if len(batch) == 2:
            x, y = batch
            cache_key = None
        elif len(batch) == 3:
            x, y, cache_key = batch
        out = self(x, cache_key=cache_key)
        loss = self.loss_fn(out, y)
        return loss, {'ce': loss}, out

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Any]], cache_key: Any = None) -> torch.Tensor:
        if self.patch_embed_caching:
            assert cache_key is not None, \
                'Cache key must be provided when patch embedding caching is enabled'

        # (B, C, H, W) -> (B, L, E)
        x = self.patch_embed(x, cache_key=cache_key)
        # (B, L, E) -> (B, C, E)
        x = self.pooling(x)
        # (B, C, E) -> (B, C, 2)
        x = self.classifier(x)
        # (B, C, 2) -> (B, 2) if C == 1 else (B, C, 2) -> (B, C, 2)
        x = x.squeeze(1)
        
        return x
    