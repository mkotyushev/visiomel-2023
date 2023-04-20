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
            nn.Linear(splitter_hidden_size, 1),
        )
    
    def __quadtree(self, patch, intra_level_index_history=None, level=0):
        # patch.shape == (1, C, H, W)
        
        if intra_level_index_history is None:
            intra_level_index_history = [0]
        
        size = patch.shape[2]
        assert size % self.patch_size == 0

        patch_preview = F.interpolate(patch, self.patch_size)
        split_logit = self.splitter(patch_preview)
        if size < self.patch_size * 2 or torch.less_equal(split_logit, 0.0):
            yield intra_level_index_history, level, patch_preview
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
                    level=level + 1
                )
        
    def forward(self, x, output_patch_index_map=False):
        B, _, H, W = x.shape
        P = self.patch_size
        assert B == 1
        assert H == W, 'Currently only square images are supported'

        # Get quad-tree embeddings
        history, levels, patches = \
            list(zip(*self.__quadtree(x)))
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
        if output_patch_index_map:
            embeddings_dense_patch_index_map = torch.zeros(
                B,
                1, 
                H // P, 
                W // P, 
                dtype=x.dtype,
                device=x.device
            )
        for i, (h, level) in enumerate(
            zip(history, levels)
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
            
            if output_patch_index_map:
                embeddings_dense_patch_index_map[
                    ..., 
                    h_start:h_start+h_size, 
                    w_start:w_start+w_size
                ] = i + 1

        if output_patch_index_map:
            return embeddings_dense, embeddings_dense_patch_index_map
        else:
            return embeddings_dense
