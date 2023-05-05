# %%
import argparse
import pandas as pd
import timm
import torch
from tqdm import tqdm

from model.patch_embed_with_backbone import PatchBackbone
from data.visiomel_datamodule import VisiomelDatamodule


# fold_index = 0
parser = argparse.ArgumentParser()
parser.add_argument('--fold-index', type=int, default=0)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--patch-batch-size', type=int, default=2)
args = parser.parse_args()

fold_index = args.fold_index

if args.aug:  # for val augmented embeddings with 5 repeats
    task = 'simmim_randaug'
    train_transform_n_repeats = 5
    val_repeats_aug = True  # to apply same repeats and augmentations as in training
    filename = 'val_aug.pkl'
else:  # for val embeddings w/o augmentations with 1 repeat
    task = 'simmim'
    train_transform_n_repeats = 1
    val_repeats_aug = False  # no augmentations and only 1 repeat
    filename = 'val.pkl'

patch_embed_backbone_name = 'swinv2_base_window12to24_192to384_22kft1k'
patch_size = 1536
patch_embed_backbone_ckpt_path = f'/workspace/visiomel-2023/weights/val_ssup_patches_aug_fold_{fold_index}/checkpoints/best.ckpt'
patch_batch_size = args.patch_batch_size
batch_size = 1
save_path = f'/workspace/visiomel-2023/weights/val_ssup_patches_aug_fold_{fold_index}/embeddings/'

# %% [markdown]
# Data

# %%
datamodule = VisiomelDatamodule(
    task=task,
    data_dir_train='/workspace/data/images_page_4_shrink/',	
    k=5,
    fold_index=fold_index,
    data_dir_test=None,
    img_size=patch_size,
    shrink_preview_scale=None,
    batch_size=batch_size,
    split_seed=0,
    num_workers=4,
    num_workers_saturated=4,
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=True,
    sampler=None,
    enable_caching=False,
    data_shrinked=True,
    train_resize_type='none',
    train_transform_n_repeats=train_transform_n_repeats,
    val_repeats_aug=val_repeats_aug,
)
datamodule.setup()
val_dataloader, _ = datamodule.val_dataloader()

# %% [markdown]
# Model

# %%
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
            .replace('model.encoder.', ''): v 
        for k, v in 
        torch.load(patch_embed_backbone_ckpt_path)['state_dict'].items()
    }
    print(backbone.load_state_dict(state_dict, strict=False))

patch_embed = PatchBackbone(
    backbone=backbone, 
    patch_size=patch_size, 
    embed_dim=backbone.num_features,
    patch_batch_size=patch_batch_size,
    patch_embed_caching=False,
).cuda().eval()

# %%
with torch.no_grad():
    features, labels, paths = [], [], []
    for batch in tqdm(val_dataloader):  # batches
        x_minibatch, mask_minibatch, y_minibatch, path_minibatch = batch
        for x, mask, y, path in zip(x_minibatch, mask_minibatch, y_minibatch, path_minibatch):  # samples
            features.append(patch_embed(x.unsqueeze(0).cuda()).detach().cpu())
            labels.append(y.detach().cpu())
            paths.append(path)

# %%
df_val = pd.DataFrame({
    'path': paths,
    'label': labels,
    'features': features,
})
df_val.to_pickle(save_path + filename)

# %%



