# 1. get checkpoint root dir: args.checkpoint_root_dir
# 2. get config path: args.config_path

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Set, Union
import pickle

import numpy as np
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.metrics import f1_score, log_loss, roc_auc_score
import torch
from tqdm import tqdm

from scripts.runs_info import get_runs_info
from utils.utils import MyLightningCLI, TrainerWandb, get_X_y_groups, oldest_checkpoint

# Log level info
import logging
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_root_dir', type=Path, default='./visiomel')
parser.add_argument('--logs_dir', type=Path, default='./wandb')
parser.add_argument('--save_path', type=Path, default='./cv_results.pkl')

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


def bootstrap_metrics(y_true: np.array, y_proba: np.array, n_bootstrap=1000, replace=True):
    neg_class_indices = np.arange(y_true.shape[0])[y_true == 0]
    pos_class_indices = np.arange(y_true.shape[0])[y_true == 1]

    metrics = defaultdict(list)
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
        metrics['log_loss_bs'].append(log_loss(y_true_downsampled, y_proba_downsampled, eps=1e-16))
        metrics['f1_score_bs'].append(f1_score(y_true_downsampled, y_proba_downsampled > 0.5))
        metrics['roc_auc_bs'].append(roc_auc_score(y_true_downsampled, y_proba_downsampled))
    return {metric_name: np.mean(metric_values) for metric_name, metric_values in metrics.items()}


# Workaround to avoid error: 'Configuration check failed :: 
# No action for destination key "ckpt_path" to check its value.'
# whenn run=False and PL-generated config file is used
sys.argv = sys.argv[:1]


class MyLightningCLINoRun(MyLightningCLI):
    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {**MyLightningCLI.subcommands(), 'no_run': {"model", "dataloaders", "datamodule"}}


class TrainerWandbNoRun(TrainerWandb):
    def no_run(
        self,
        model: Optional[LightningModule] = None,
        dataloaders: Optional[Union[Any, LightningDataModule]] = None,
        datamodule: Optional[LightningDataModule] = None,
        return_predictions: Optional[bool] = None,
        ckpt_path: Optional[str] = None,
    ):
        pass


n_bootstrap = 1000

cv_results = defaultdict(dict)
for fold_index_test in tqdm(range(5)):
    # PAC model
    data = defaultdict(dict)
    for fold_index in tqdm(range(5)):
        cli = MyLightningCLINoRun(
            trainer_class=TrainerWandbNoRun, 
            save_config_kwargs={
                'config_filename': 'config_pl.yaml',
                'overwrite': True,
            },
            args=[
                'no_run',
                '--config', str(fold_to_ckpt_info[fold_index_test][fold_index]['config_path']),
            ],
            run=True,
        )

        data['pac'][fold_index] = {}
        # Predict on val
        cli.datamodule.setup()
        y_proba = torch.softmax(
            torch.concat(
                cli.trainer.predict(
                    model=cli.model,
                    dataloaders=cli.datamodule.val_dataloader()[0],  # only not downsampled
                    return_predictions=True,
                    ckpt_path=fold_to_ckpt_info[fold_index_test][fold_index]['ckpt_path'],
                ), 
                dim=0
            ).float(), 
            dim=1
        ).numpy()
        _, y_val, _ = get_X_y_groups(cli.datamodule.val_dataset.data)
        data['pac'][fold_index]['val_metrics'] = {
            'log_loss': log_loss(y_val, y_proba[:, 1], eps=1e-16),
            'f1_score': f1_score(y_val, y_proba[:, 1] > 0.5),
            'roc_auc': roc_auc_score(y_val, y_proba[:, 1]),
        }

        # Predict on test
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
        data['pac'][fold_index]['y_proba_test'] = y_proba[:, 1]

    # Mean test predictions over folds
    data['pac'][-1] = np.mean(np.stack(list([data['pac'][i]['y_proba_test'] for i in range(5)])), axis=0)

    # GT
    X_test, y_test, _ = get_X_y_groups(cli.datamodule.test_dataset.data)
    data['gt'][-1] = y_test

    # Metrics
    # Log loss
    cv_results[fold_index_test]['log_loss'] = log_loss(data['gt'][-1], data['pac'][-1], eps=1e-16)

    # F1 score
    cv_results[fold_index_test]['f1_score'] = f1_score(
        data['gt'][-1],
        data['pac'][-1] > 0.5,
    )

    # ROC AUC
    cv_results[fold_index_test]['roc_auc'] = roc_auc_score(
        data['gt'][-1],
        data['pac'][-1],
    )

    # Bootstrap metrics on n_bootstrap test sets with downsampled negative class
    bootstrap_metrics_dict = bootstrap_metrics(
        data['gt'][-1], 
        data['pac'][-1],
        n_bootstrap=n_bootstrap,
        replace=True,
    )
    for metric_name, metric_value in bootstrap_metrics_dict.items():
        cv_results[fold_index_test][metric_name] = metric_value

    # Val metrics
    for metric_name in data['pac'][0]['val_metrics']:
        cv_results[fold_index_test][metric_name + '_val'] = sum(data['pac'][i]['val_metrics'][metric_name] for i in range(5)) / 5

    # Raw data
    cv_results[fold_index_test]['data'] = data

# Print results
print('========================================')
for fold_index_test, fold_results in cv_results.items():
    print(f'Fold {fold_index_test}')
    for metric_name, metric_value in fold_results.items():
        if metric_name == 'data':
            continue
        print(f'\t{metric_name}: {metric_value}')
    print()

print('========================================')
for metric_name in cv_results[0].keys():
    if metric_name == 'data':
        continue
    print(f'{metric_name}: {np.mean([fold_results[metric_name] for fold_results in cv_results.values()])}')

# Save results to file in pickle format
with open(args.save_path, 'wb') as f:
    pickle.dump(cv_results, f)
