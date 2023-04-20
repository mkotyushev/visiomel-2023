
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed
from timm.models.swin_transformer_v2 import SwinTransformerV2

from model.quadtree_embedding import QuadtreeEmbedding


class PatchBackbone(nn.Module):
    """Apply backbone network to each patch."""
    def __init__(self, backbone, patch_size=16, embed_dim=768):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        if backbone.num_features != embed_dim:
            self.linear = nn.Linear(backbone.num_features, embed_dim)
        else:
            self.linear = nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Crop to patch size multiplier as in PatchEmbed
        x = x[:, :H // P * P, :W // P * P]
        
        # Extract patches
        x = x \
            .unfold(2, P, P) \
            .unfold(3, P, P)
        
        # Concat to batch dimension
        # (B, C, H_patches, W_patches, patch_size, patch_size) -> 
        # (B * H_patches * W_patches, C, patch_size, patch_size)
        H_patches, W_patches = x.shape[2:4]
        x = x.reshape(B * H_patches * W_patches, C, P, P)

        # Apply backbone network
        x = self.backbone(x)

        # Map to required embedding dimension
        assert x.ndim == 2
        x = self.linear(x)

        # Reshape back as expected from PatchEmbed
        # (B * H_patches * W_patches, *backbone_out_shape) ->
        # (B, H * W, *backbone_out_shape)
        x = x.reshape(B, H_patches * W_patches, *x.shape[1:])

        return x
    

class PatchBackboneQuadtree(nn.Module):
    """Apply backbone network to each patch selected by quadtree."""
    def __init__(self, backbone, patch_size=16, embed_dim=768, splitter_hidden_size=64):
        super().__init__()
        self.embedding = QuadtreeEmbedding(
            backbone, 
            splitter_hidden_size=splitter_hidden_size, 
            patch_size=patch_size
        )
        self.patch_size = patch_size
        if backbone.num_features != embed_dim:
            self.linear = nn.Linear(backbone.num_features, embed_dim)
        else:
            self.linear = nn.Identity()
    
    def forward(self, x):
        B = x.shape[0]

        # Apply embedding extractor object-wise
        # (B, C, H, W) ->
        # (B, backbone_emb_dim, H_patches, W_patches)
        x = torch.cat(
            [self.embedding(x[i].unsqueeze(0)) for i in range(B)],
            0
        )
        H_patches, W_patches = x.shape[-2:]

        # Swap dims 
        # (B, backbone_emb_dim, H_patches, W_patches) ->
        # (B, H_patches, W_patches, backbone_emb_dim)
        x = x.permute(0, 2, 3, 1)

        # Map to required embedding dimension
        # (B, H_patches, W_patches, backbone_emb_dim) ->
        # (B, H_patches, W_patches, emb_dim) ->
        x = self.linear(x)

        # Reshape back as expected from PatchEmbed
        # (B, H_patches, W_patches, emb_dim) ->
        # (B, H * W, emb_dim)
        x = x.reshape(B, H_patches * W_patches, x.shape[-1])

        return x


class PatchEmbedWithBackbone(PatchEmbed):
    """ 2D Image to Patch Embedding.
    
    Each patch will be projected to a vector 
    via backbone network.
    """
    def __init__(
        self,
        backbone,
        quadtree=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=False,  # Already flattened by backbone
            bias=bias
        )
        if not quadtree:
            self.proj = PatchBackbone(backbone, patch_size, embed_dim)
        else:
            self.proj = PatchBackboneQuadtree(backbone, patch_size, embed_dim)


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
            self, patch_embed_backbone: nn.Module, quadtree: bool = False,
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
            backbone=patch_embed_backbone, quadtree=quadtree,
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
