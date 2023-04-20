import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadtreeEmbedding(nn.Module):
    def __init__(self, backbone, splitter_hidden_size=64, patch_size=64):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.splitter = nn.Sequential(
            nn.Conv2d(3, splitter_hidden_size // 4, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(splitter_hidden_size // 4),
            nn.Conv2d(splitter_hidden_size // 4, splitter_hidden_size // 2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(splitter_hidden_size // 2),
            nn.Conv2d(splitter_hidden_size // 2, splitter_hidden_size, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(splitter_hidden_size),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(splitter_hidden_size, 2),
        )
    
    def __quadtree(self, patch, intra_level_index_history=None, level=0, random_split=False):
        # patch.shape == (1, C, H, W)
        
        if intra_level_index_history is None:
            intra_level_index_history = [0]
        
        size = patch.shape[2]
        assert size % self.patch_size == 0

        patch_preview = F.interpolate(patch, self.patch_size)
        if random_split:
            split_logit = need_split = ((torch.rand(1, device=patch.device) - 0.5) <= 0.0).long()
        else:
            split_logit = self.splitter(patch_preview)
            need_split = torch.greater_equal(split_logit[:, 1], split_logit[:, 0])
        
        if size < self.patch_size * 2 or not need_split:
            yield split_logit, intra_level_index_history, level, patch_preview
        else:
            sub_patches = \
                patch[..., :patch.shape[-2] // 2, :patch.shape[-1] // 2], \
                patch[..., :patch.shape[-2] // 2, patch.shape[-1] // 2:], \
                patch[..., patch.shape[-2] // 2:, :patch.shape[-1] // 2], \
                patch[..., patch.shape[-2] // 2:, patch.shape[-1] // 2:]
            for i, sub_patch in enumerate(sub_patches):
                yield from self.__quadtree(
                    sub_patch, 
                    intra_level_index_history=intra_level_index_history + [i], 
                    level=level + 1,
                    random_split=random_split
                )
        
    def forward(self, x, output_split_logits=False, random_split=False):
        B, _, H, W = x.shape
        P = self.patch_size
        assert B == 1
        assert H == W, 'Currently only square images are supported'

        # Get quad-tree embeddings
        split_logits, history, levels, patches = \
            list(zip(*self.__quadtree(x, random_split=random_split)))
        embeddings_sparse = self.backbone(torch.cat(patches, 0))  # n_patches, emb_dim
        embeddings_sparse = embeddings_sparse.unsqueeze(-1).unsqueeze(-1)  # n_patches, emb_dim, 1, 1

        # Repeat in quads
        embeddings_dense = torch.zeros(
            B,
            embeddings_sparse.shape[1], 
            H // P, 
            W // P, 
            dtype=x.dtype,
            device=x.device
        )
        if output_split_logits:
            split_logits_map = torch.zeros(
                B,
                1, 
                H // P, 
                W // P, 
                split_logits[0].ndim,
                dtype=split_logits[0].dtype,
                device=x.device
            )
        for i, (split_logit, h, level) in enumerate(
            zip(split_logits, history, levels)
        ):            
            h_start, w_start = 0, 0
            for l, intra_level_index in enumerate(h):
                h_size, w_size = \
                    H // P // (2 ** l), \
                    W // P // (2 ** l)
                
                if intra_level_index == 0:
                    pass
                elif intra_level_index == 1:
                    w_start += w_size
                elif intra_level_index == 2:
                    h_start += h_size
                elif intra_level_index == 3:
                    h_start += h_size
                    w_start += w_size

            h_size, w_size = \
                H // P // (2 ** level), \
                W // P // (2 ** level)

            embeddings_dense[
                ..., 
                h_start:h_start+h_size, 
                w_start:w_start+w_size
            ] = \
                embeddings_sparse[i, ...].expand(-1, h_size, -1).expand(-1, -1, w_size)
            
            if output_split_logits:
                split_logits_map[
                    ..., 
                    h_start:h_start+h_size, 
                    w_start:w_start+w_size, 
                    :
                ] = split_logit

        if output_split_logits:
            return embeddings_dense, split_logits_map
        else:
            return embeddings_dense
