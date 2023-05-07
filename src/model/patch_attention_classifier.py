import pandas as pd
import timm
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union

from model.patch_embed_with_backbone import PatchBackbone
from model.visiomel_model import VisiomelClassifier


"""
================================================ age              ================================================
    66    80
    64    78
    62    71
    70    70
    60    62
    72    62
    78    58
    68    56
    58    54
    46    48
    56    48
    74    44
    80    43
    50    42
    40    41
    52    41
    82    41
    76    38
    42    37
    54    34
    44    33
    48    32
    86    31
    84    30
    38    25
    36    22
    32    17
    30    15
    90    15
    34    14
    88    13
    26    10
    28     9
    18     8
    92     7
    24     4
    94     4
    22     2
    20     1
    98     1
    16     1

    >>> pd.qcut(df['age'].apply(lambda x: int(x[1:3])), q=4).value_counts()
    (15.999, 50.0]    361
    (50.0, 63.0]      310
    (63.0, 72.0]      346
    (72.0, 98.0]      325
    Name: age, dtype: int64
    >>> pd.cut(df['age'].apply(lambda x: int(x[1:3])), bins=[-10.0, 50.0, 63.0, 72.0, 1000.0]).value_counts()
    (-10.0, 50.0]     361
    (50.0, 63.0]      310
    (63.0, 72.0]      346
    (72.0, 1000.0]    325
    Name: age, dtype: int64

================================================ sex              ================================================
    1    677
    2    665

    relapse    0    1
    sex              
    1        562  115
    2        567   98
================================================ melanoma_history ================================================
    >>> df['melanoma_history'].fillna('UNK').value_counts()
    UNK    700
    NO     585
    YES     57

    relapse             0    1
    melanoma_history          
    NO                451  134
    UNK               637   63
    YES                41   16
================================================ body_site        ================================================

    Values counts for `body_site` categorical feature on train dataset:

    0 = upper: mask = (df.body_site == 'upper limb/shoulder') | (df.body_site == 'arm') | (df.body_site == 'forearm') | (df.body_site == 'hand') | (df.body_site == 'hand/foot/nail') | (df.body_site == 'nail') | (df.body_site == 'finger')
        relapse                         0       1       ratio
                                        238     43      5.534883720930233
        upper limb/shoulder     70
        arm                    149
        forearm                 26
        hand                     7
        hand/foot/nail          21
        nail                     2
        finger                   6
        other                    ?

    1 = center: mask = (df.body_site == 'trunc') | (df.body_site == 'trunk')
        relapse                         0       1       ratio
                                        378     60      6.3
        trunc                  306
        trunk                  132

    2 = head: mask = (df.body_site == 'head/neck') | (df.body_site == 'face') | (df.body_site == 'neck') | (df.body_site == 'scalp')
        relapse                         0       1       ratio
                                        180     41      4.390243902439025
        head/neck               71
        face                   134
        neck                    12
        scalp                    4
        
    3 = lower: mask = (df.body_site == 'leg') | (df.body_site == 'lower limb/hip') | (df.body_site == 'thigh') | (df.body_site == 'foot') | (df.body_site == 'toe') | (df.body_site == 'sole') | (df.body_site == 'seat')
        relapse                         0       1       ratio
                                        319     65      4.907692307692308
        leg                    163
        lower limb/hip         114
        thigh                   58
        foot                    23
        toe                      7
        sole                    13
        seat                     6

    Relapse values counts for each `body_site` categorical feature 
    value on train dataset:
        relapse                0   1
        body_site                   
        arm                  138  11
        face                 119  15
        finger                 5   1
        foot                  15   8
        forearm               21   5
        hand                   7   0
        hand/foot/nail        11  10
        head/neck             50  21
        leg                  146  17
        lower limb/hip        90  24
        nail                   1   1
        neck                  10   2
        scalp                  1   3
        seat                   4   2
        sole                   7   6
        thigh                 50   8
        toe                    7   0
        trunc                275  31
        trunk                103  29
        upper limb/shoulder   55  15


"""

class PatchAttentionPooling(nn.Module):
    def __init__(self, n_classes, embed_dim=1536, hidden_dim=64, num_heads=1, dropout=0.2):
        super().__init__()

        # Class
        self.class_emb = nn.Parameter(torch.randn(1, n_classes, hidden_dim))

        # Meta

        # pandas qcut with 4 bins
        # ~ [-10.0, 50.0, 63.0, 72.0, 1000.0] on train dataset
        # unknown in the firsts interval with 0.0
        self.age_emb = nn.Parameter(torch.randn(1, 4, embed_dim))

        # 0 + unknown, 1
        self.sex_emb = nn.Parameter(torch.randn(1, 2, embed_dim))

        # some of 20 body sites are underrepresented in the train dataset
        # and some are essentially the same, so group it (see stats above)
        # 0 = upper and unknown, 1 = center, 2 = head, 3 = lower
        self.body_site_emb = nn.Parameter(torch.randn(1, 4, embed_dim))

        # 0 = no, 1 = yes, 2 = unknown
        self.melanoma_history_emb = nn.Parameter(torch.randn(1, 3, embed_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            kdim=embed_dim,
            vdim=embed_dim,
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: torch.LongTensor = None, meta: torch.Tensor = None):
        # x: (B, L, E)
        # mask: (B, L)
        # meta: (B, 4)
        B = x.shape[0]
        query = self.class_emb.repeat(B, 1, 1)

        # prepend meta features to x and ones to mask if any
        if meta is not None:
            # meta[:, 0] - age 0, 1, 2 or 3
            # meta[:, 1] - sex 0 or 1, index of sex embedding
            # meta[:, 2] - body_site 0, 1, 2 or 3, index of body site embedding
            # meta[:, 3] - melanoma_history 0, 1 or 2, index of melanoma history embedding
            age_emb = self.age_emb[:, meta[:, 0]].permute(1, 0, 2)  # (B, 1, E)
            sex_emb = self.sex_emb[:, meta[:, 1]].permute(1, 0, 2)  # (B, 1, E)
            body_site_emb = self.body_site_emb[:, meta[:, 2]].permute(1, 0, 2)  # (B, 1, E)
            melanoma_history_emb = self.melanoma_history_emb[:, meta[:, 3]].permute(1, 0, 2)  # (B, 1, E)

            x = torch.cat((age_emb, sex_emb, body_site_emb, melanoma_history_emb, x), dim=1)  # (B, 4 + L, E)
            if mask is not None:
                mask = torch.cat((torch.ones(B, 4, dtype=torch.bool, device=mask.device), mask), dim=1)

        # key and value are the same, so mask should be
        # the same for both
        attn_mask = mask.unsqueeze(1).repeat(self.attention.num_heads, 1, 1)

        x, _ = self.attention(query, x, x, attn_mask=attn_mask, key_padding_mask=mask)
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
        emb_precalc: bool = False,
        emb_precalc_dim: int = 1024,
        label_smoothing: float = 0.0,
        lr: float = 1e-3,
        attention_num_heads: int = 2,
        attention_dropout: float = 0.2,
        n_bootstrap: int = 1000,
    ):
        # Hack to make wandb CV work with nested dicts
        optimizer_init['init_args']['lr'] = lr
        
        super().__init__(
            optimizer_init=optimizer_init,
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning,
            log_norm_verbose=log_norm_verbose,
            lr_layer_decay=lr_layer_decay,
            label_smoothing=label_smoothing,
            n_bootstrap=n_bootstrap,
        )
        self.save_hyperparameters()

        if not emb_precalc:
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
        else:
            self.patch_embed = None
        self.pooling = PatchAttentionPooling(
            n_classes=num_classes if num_classes > 2 else 1, 
            embed_dim=emb_precalc_dim,
            hidden_dim=attention_hidden_dim,
            num_heads=attention_num_heads,
            dropout=attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(attention_hidden_dim, attention_hidden_dim // 2),
            nn.LeakyReLU(),
            nn.LayerNorm(attention_hidden_dim // 2),
            nn.Linear(attention_hidden_dim // 2, attention_hidden_dim // 4),
            nn.LeakyReLU(),
            nn.LayerNorm(attention_hidden_dim // 4),
            nn.Linear(attention_hidden_dim // 4, 2),
        )

        self.patch_embed_caching = patch_embed_caching

        self.unfreeze_only_selected()
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        x, mask, meta, y, cache_key = None, None, None, None, None
        if len(batch) == 2:
            x, y = batch
            cache_key = None
        elif len(batch) == 3:
            x, y, cache_key = batch
        elif len(batch) == 4:
            x, mask, y, cache_key = batch
        elif len(batch) == 5:
            x, mask, meta, y, cache_key = batch
        out = self(x, mask, meta, cache_key=cache_key)
        loss = self.loss_fn(out, y)
        return loss, {'ce': loss}, out

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Any]], mask: torch.BoolTensor = None, meta: torch.Tensor = None, cache_key: Any = None) -> torch.Tensor:
        if self.patch_embed is not None:
            if self.patch_embed_caching:
                assert cache_key is not None, \
                    'Cache key must be provided when patch embedding caching is enabled'
            # (B, C, H, W) -> (B, L, E)
            x = self.patch_embed(x, cache_key=cache_key)

        # x: (B, L, E), meta: (B, 3) -> (B, 1, E)
        x = self.pooling(x, mask=mask, meta=meta)
        # (B, 1, E) -> (B, E)
        x = x.squeeze(1)
        # (B, E) -> (B, 2)
        x = self.classifier(x)
        
        return x
    
    def update_metrics(self, span, preds, batch):
        """Update train metrics."""
        if len(batch) == 4:  # no meta: x, mask, y, path
            y, y_pred = batch[2].detach(), preds[:, 1].detach().float()
        elif len(batch) == 5:  # with meta: x, mask, meta, y, path
            y, y_pred = batch[3].detach(), preds[:, 1].detach().float()
        self.cat_metrics[span]['preds'].update(y_pred)
        self.cat_metrics[span]['targets'].update(y)
