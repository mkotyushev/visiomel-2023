# 1. get checkpoint root dir: args.checkpoint_root_dir
# 2. get config path: args.config_path

import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, log_loss
import torch
from tqdm import tqdm

from scripts.runs_info import get_runs_info
from utils.utils import MyLightningCLI, TrainerWandb, get_X_y_groups, oldest_checkpoint

# Log level info
import logging
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_root_dir', type=str, default='./visiomel')
parser.add_argument('--logs_dir', type=str, default='./wandb')

args = parser.parse_args()

# Get oldest checkpoint for each nested fold
runs_info = get_runs_info(args.checkpoint_root_dir, args.logs_dir)
fold_to_ckpt_info = defaultdict(dict)
for run_info in runs_info.values():
    fold_to_ckpt_info[run_info['fold_index_test']][run_info['fold_index']] = {
        'ckpt_path': oldest_checkpoint(
            run_info['checkpoint_paths']
        ),
        'config_path': run_info['config_path'],
    }

logging.info(f'Found {len(fold_to_ckpt_info)} nested folds')
for fold_index_test, fold_info in fold_to_ckpt_info.items():
    logging.info(
        f'Found {len(fold_info)} folds for nested fold {fold_index_test}, '
        f'missing folds: {set(range(5)) - set(fold_info.keys())}'
    )


def bootstrap_log_loss(y_true: np.array, y_proba: np.array, n_bootstrap=1000, replace=True):
    neg_class_indices = np.arange(y_true.shape[0])[y_true == 0]
    pos_class_indices = np.arange(y_true.shape[0])[y_true == 1]

    log_losses = []
    for _ in range(n_bootstrap):
        # Downsample y_true and y_proba negative class
        # keep all positive class
        downsampled_neg_class_indices = np.random.choice(
            neg_class_indices,
            size=pos_class_indices.shape[0],
            replace=replace,
        )
        y_true_downsampled = np.concatenate(
            [
                y_true[pos_class_indices],
                y_true[downsampled_neg_class_indices],
            ],
            axis=0,
        )
        y_proba_downsampled = np.concatenate(
            [
                y_proba[pos_class_indices],
                y_proba[downsampled_neg_class_indices],
            ],
            axis=0,
        )
        log_losses = log_loss(y_true_downsampled, y_proba_downsampled)
    return np.mean(log_losses)


n_bootstrap = 1000

cv_results = defaultdict(dict)
for fold_index_test in tqdm(range(5)):
    # PAC model
    data = defaultdict(dict)
    for fold_index in tqdm(range(5)):
        cli = MyLightningCLI(
            trainer_class=TrainerWandb, 
            save_config_kwargs={
                'config_filename': 'config_pl.yaml',
                'overwrite': True,
            },
            args=[
                '--config', fold_to_ckpt_info[fold_index_test][fold_index]['config_path'],
            ],
            run=False,
        )

        # Patch attention model
        y_proba = torch.softmax(
            torch.concat(
                cli.trainer.predict(
                    model=cli.model,
                    datamodule=cli.datamodule, 
                    return_predictions=True,
                    ckpt_path=fold_to_ckpt_info[fold_index_test][fold_index]['ckpt_path'],
                ), 
                dim=0
            ).float(), 
            dim=1
        ).numpy()
        data['pac'][fold_index] = y_proba[:, 1]

    # Mean over folds
    data['pac'][-1] = np.mean(np.stack(list(data['pac'].values())), axis=0)

    # GT
    X_test, y_test, _ = get_X_y_groups(cli.datamodule.test_dataset.data)
    data['gt'][-1] = y_test

    # Metrics
    # Log loss on test set
    cv_results[fold_index_test]['log_loss'] = log_loss(data['gt'][-1], data['pac'][-1])

    # Log loss bootstrap of n_bootstrap test sets with downsampled negative class
    cv_results[fold_index_test]['log_loss_bootstrap'] = bootstrap_log_loss(
        data['gt'][-1], 
        data['pac'][-1],
        n_bootstrap=n_bootstrap,
        replace=True,
    )

    # F1 score
    cv_results[fold_index_test]['f1_score'] = f1_score(
        data['gt'][-1],
        data['pac'][-1] > 0.5,
    )
