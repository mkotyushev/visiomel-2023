
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch
from timm.models.layers import PatchEmbed
from timm.models.swin_transformer_v2 import SwinTransformerV2
from data.transforms import generate_tensor_patches

from model.quadtree_embedding import QuadtreeEmbedding
from model.drloc.aux_modules import DenseRelativeLoc


def build_drloc(params):
    drloc = nn.ModuleList()
    if params['use_multiscale']:
        for i_layer in range(params['num_layers']):
            drloc.append(DenseRelativeLoc(
                in_dim=min(int(params['embed_dim'] * 2 ** (i_layer+1)), params['num_features']), 
                out_dim=2 if params['drloc_mode']=="l1" else max(params['img_size'] // (4 * 2**i_layer), params['img_size']//(4 * 2**(params['num_layers']-1))),
                sample_size=params['sample_size'],
                drloc_mode=params['drloc_mode'],
                use_abs=params['use_abs']))
    else:
        drloc.append(DenseRelativeLoc(
            in_dim=params['num_features'], 
            out_dim=2 if params['drloc_mode']=="l1" else params['img_size']//(4 * 2**(params['num_layers']-1)),
            sample_size=params['sample_size'],
            drloc_mode=params['drloc_mode'],
            use_abs=params['use_abs']))
    return drloc


class PatchBackbone(nn.Module):
    """Apply backbone network to each patch."""
    def __init__(self, backbone, patch_size=16, embed_dim=768, patch_batch_size=1, fill=0, patch_embed_caching=False):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        if backbone.num_features != embed_dim:
            self.linear = nn.Linear(backbone.num_features, embed_dim)
        else:
            self.linear = nn.Identity()
        self.patch_batch_size = patch_batch_size
        self.fill = fill

        self.cache = None
        if patch_embed_caching:
            self.cache = {}
    
    def forward(self, x, cache_key=None):
        if self.cache is not None and cache_key is not None:
            result = []
            for i, key in enumerate(cache_key):
                if key not in self.cache:
                    self.cache[key] = self._forward(x[i].unsqueeze(0)).cpu()
                result.append(self.cache[key].to(x.device))
            result = torch.cat(result, dim=0)
            return result
        else:
            return self._forward(x)
    
    def _forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Extract patches & apply backbone network
        iterator = generate_tensor_patches(x, (P, P), fill=self.fill)
        x_patch_batches_embedded = []
        try:
            while True:
                x_patch_batch = []
                for _ in range(self.patch_batch_size):
                    x_patch_batch.append(next(iterator))
                x_patch_batch = torch.cat(x_patch_batch, dim=0)
                x_patch_batches_embedded.append(self.backbone(x_patch_batch))
        except StopIteration:
            if len(x_patch_batch) > 0:
                x_patch_batch = torch.cat(x_patch_batch, dim=0)
                x_patch_batches_embedded.append(self.backbone(x_patch_batch))
        x = torch.cat(x_patch_batches_embedded, dim=0)

        # Map to required embedding dimension
        assert x.ndim == 2
        x = self.linear(x)

        # Reshape back as expected from PatchEmbed
        # (B * H_patches * W_patches, *backbone_out_shape) ->
        # (B, H * W, *backbone_out_shape)
        x = x.reshape(B, -1, *x.shape[1:])

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


class SwinTransformerV2Modded(SwinTransformerV2):
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
            self, patch_embed_backbone: Optional[nn.Module] = None, quadtree: bool = False,
            img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            window_size=7, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            pretrained_window_sizes=(0, 0, 0, 0), drloc_params=None, **kwargs):
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

        if patch_embed_backbone is not None:
            self.patch_embed = PatchEmbedWithBackbone(
                backbone=patch_embed_backbone, quadtree=quadtree,
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)

        self.drloc_params = drloc_params
        if drloc_params is not None:
            drloc_params['num_layers'] = len(depths)
            drloc_params['num_features'] = int(embed_dim * 2 ** (len(depths) - 1))
            drloc_params['embed_dim'] = embed_dim
            drloc_params['img_size'] = img_size
            self.drloc = build_drloc(drloc_params)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if self.drloc_params is not None and self.drloc_params['use_multiscale']:
            all_layers = []
            for layer in self.layers:
                x = layer(x)
                all_layers.append(x)
            return all_layers
        else:
            for layer in self.layers:
                x = layer(x)
            return [x]

    def forward(self, x):
        x_layers = self.forward_features(x)
        x = self.norm(x_layers[-1])
        x = self.forward_head(x)

        if self.drloc is None:
            return x

        outs = Munch(sup=x)
        outs.drloc = []
        outs.deltaxy = []
        outs.plz = []

        for idx, x_cur in enumerate(x_layers):
            x_cur = x_cur.transpose(1, 2) # [B, C, L]
            B, C, HW = x_cur. size()
            H = W = int(math.sqrt(HW))
            feats = x_cur.view(B, C, H, W) # [B, C, H, W]

            drloc_feats, deltaxy = self.drloc[idx](feats)
            outs.drloc.append(drloc_feats)
            outs.deltaxy.append(deltaxy)
            outs.plz.append(H) # plane size 
        return outs