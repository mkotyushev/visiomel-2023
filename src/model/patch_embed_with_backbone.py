
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed


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


class PatchEmbedWithBackbone(PatchEmbed):
    """ 2D Image to Patch Embedding.
    
    Each patch will be projected to a vector 
    via backbone network.
    """
    def __init__(
        self,
        backbone=None,
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
        if backbone is not None:
            self.proj = PatchBackbone(backbone, patch_size, embed_dim)
