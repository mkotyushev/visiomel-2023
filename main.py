import os
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import argparse
import pandas as pd
import timm
import torch
from pytorch_lightning import seed_everything
from collections import defaultdict
from joblib import load
from unittest.mock import patch
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None

# register swin transformer
from lib.timm_cherrypick.swin_transformer_v2 import *
from src.data.visiomel_datamodule import VisiomelDatamodule
from src.data.visiomel_datamodule_emb import VisiomelDatamoduleEmb
from src.model.patch_attention_classifier import PatchBackbone, PatchAttentionClassifier
from src.utils.utils import get_X_y_groups, find_classes_mock, make_dataset_mock


solution_to_params = {
    'val_ll': [
        {
            'datamodule': {
                'meta_filepath': None,
            },
            'pac': {
                'attention_num_heads': 8,
                'attention_hidden_dim': 64,
                'state_dict_dir': './weights-inference/val-ll'
            }
        },
    ],
    'val_ds_ll': [
        {
            'datamodule': {
                'meta_filepath': './data/test_metadata.csv',
            },
            'pac': {
                'attention_num_heads': 1,
                'attention_hidden_dim': 32,
                'state_dict_dir': './weights-inference/val-ds-ll'
            }
        },
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--solution', type=str, choices=['val_ll', 'val_ds_ll'], default='val_ll')
    args = parser.parse_args()

    seed_everything(0)
    # ================================================================
    #                               SSUP
    # ================================================================
 
    # Load SSUP model
    task = 'simmim'
    train_transform_n_repeats = 1
    val_repeats_aug = False  # no augmentations and only 1 repeat

    patch_embed_backbone_name = 'swinv2_base_window12to24_192to384_22kft1k'
    patch_size = 1536
    patch_embed_backbone_state_dict_path = f'./weights-inference/simmim.pth'
    patch_batch_size = 1
    batch_size = 1

    backbone = timm.create_model(
        patch_embed_backbone_name, 
        img_size=patch_size, 
        pretrained=False, 
        num_classes=0
    )
    # If backbone is fine-tuned then it is done via SwinTransformerV2SimMIM
    # module, so we need to remove the prefix 'model.encoder.' from the
    # checkpoint state_dict keys.
    state_dict = {
        k \
            .replace('model.encoder.', ''): v 
        for k, v in 
        torch.load(patch_embed_backbone_state_dict_path).items()
    }
    backbone.load_state_dict(state_dict, strict=False)
    patch_embed = PatchBackbone(
        backbone=backbone, 
        patch_size=patch_size, 
        embed_dim=backbone.num_features,
        patch_batch_size=patch_batch_size,
        patch_embed_caching=False,
    ).cuda().eval()
    
    # Data 
    # Note: test has no proper dir structure, so we need to mock 
    # some internals of ImageFolder
    with \
        patch('torchvision.datasets.folder.find_classes', find_classes_mock), \
        patch('torchvision.datasets.folder.make_dataset', make_dataset_mock):
        datamodule = VisiomelDatamodule(
            task=task,
            data_dir_train=args.data_path,	
            data_dir_test=args.data_path,	
            k=None,
            fold_index=0,
            img_size=patch_size,
            shrink_preview_scale=8,
            batch_size=batch_size,
            split_seed=0,
            num_workers=0,
            num_workers_saturated=0,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=None,
            sampler=None,
            enable_caching=False,
            data_shrinked=False,
            train_resize_type='none',
            train_transform_n_repeats=train_transform_n_repeats,
            val_repeats_aug=val_repeats_aug,
        )
        datamodule.setup()
    test_dataloader = datamodule.test_dataloader()

    # Generate embeddings 
    with torch.no_grad():
        features, labels, paths = [], [], []
        for batch in tqdm(test_dataloader):  # batches
            x_minibatch, _, y_minibatch, path_minibatch = batch
            for x, y, path in zip(x_minibatch, y_minibatch, path_minibatch):  # samples
                features.append(patch_embed(x.unsqueeze(0).cuda()).detach().cpu())
                labels.append(y.detach().cpu())
                paths.append(path)
    
    df_test = pd.DataFrame({
        'path': paths,
        'label': labels,  # fake all 0 labels here due to mock
        'features': features,
    })

    # Save embeddings to disk to have unified inference with
    # training via VisiomelDatamoduleEmb
    df_test.to_pickle('df_test.pkl')
    # df_test = pd.read_pickle('df_test.pkl')  # for debug
    
    # ================================================================
    #                               SUP
    # ================================================================
    data = defaultdict(list)

    for solution_index in range(len(solution_to_params[args.solution])):
        datamodule = VisiomelDatamoduleEmb(
            embedding_pathes=['df_test.pkl'],	
            embedding_pathes_aug_with_repeats=['df_test.pkl'],	
            batch_size=32,
            k=None,
            fold_index=0,
            k_test=None,
            fold_index_test=0,
            split_seed=0,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=False,
            sampler=None,
            num_workers_saturated=0,
            meta_filepath=solution_to_params[args.solution][solution_index]['datamodule']['meta_filepath'],
        )
        datamodule.setup()

        # lr, label_smoothing, attention_dropout etc. are not used
        # during inference, so we can pass any values here
        model = PatchAttentionClassifier(
            num_classes=2,
            patch_embed_backbone_name='swinv2_base_window12to24_192to384_22kft1k',
            patch_embed_backbone_ckpt_path=None,
            patch_size=1536,
            patch_batch_size=32,
            optimizer_init={'init_args': {'lr': 0.0008549720251132047}},
            lr_scheduler_init=None,
            pl_lrs_cfg=None,
            finetuning=None,
            log_norm_verbose=False,
            lr_layer_decay=1.0,
            grad_checkpointing=False,
            patch_embed_caching=False,
            emb_precalc=True,
            emb_precalc_dim=1024,
            label_smoothing=0.19365209170598277,
            lr=0.0008549720251132047,
            attention_dropout=0.1101479509435598,
            attention_num_heads=solution_to_params[args.solution][solution_index]['pac']['attention_num_heads'],
            attention_hidden_dim=solution_to_params[args.solution][solution_index]['pac']['attention_hidden_dim'],
        )
        model = model.cuda().eval()

        # Patch attention classifier model
        for state_dict_path in Path(
            solution_to_params[args.solution][solution_index]['pac']['state_dict_dir']
        ).glob('**/*.pth'):
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)

            pathes, y_logits = [], []
            with torch.no_grad():
                for batch in datamodule.test_dataloader():
                    x_batch, mask_batch, _, path_batch = batch
                    y_logits.append(model(x_batch.cuda(), mask_batch.cuda()).detach().cpu())
                    pathes.extend(path_batch)
            y_proba = torch.softmax(torch.cat(y_logits, dim=0), dim=1)
            
            data['pac'].append(y_proba[:, 1])
    data['path'].extend(pathes)  # same for all fold, so take last one

    # Ensemble
    df = pd.DataFrame(
        {
            'pac': sum(data['pac']) / len(data['pac']),
            'path': data['path'],
        }
    )
    df_test = pd.merge(df_test, df, on='path', how='left')
    df_test['relapse'] = df_test['pac']

    # Save submission
    df_test['filename'] = df_test['path'].apply(lambda x: x.split('/')[-1])

    df_test_format = pd.read_csv(f'{args.data_path}/submission_format.csv')
    df_test = pd.merge(df_test_format, df_test, on='filename', how='left')
    # assert (df_test['filename'].values == df_test_format['filename'].values).all()
    df_test['relapse'] = df_test['relapse_y']
    df_test[['filename', 'relapse']].to_csv('submission.csv', index=False)
    

if __name__ == '__main__':
    main()
