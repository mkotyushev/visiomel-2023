# 1. get checkpoint root dir: args.checkpoint_root_dir
# 2. get config path: args.config_path
# usage: python evaluate_nested_cv_pac.py --checkpoint_root_dir ./visiomel --logs_dir ./wandb --nested --save_path ./cv_results.pkl

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


def print_results(cv_results):
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


# Workaround to avoid error: 'Configuration check failed :: 
# No action for destination key "ckpt_path" to check its value.'
# whenn run=False and PL-generated config file is used
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_root_dir', type=Path, default='./visiomel')
    parser.add_argument('--logs_dir', type=Path, default='./wandb')
    parser.add_argument('--nested', action='store_true')
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

    if args.nested:
        assert len(fold_to_ckpt_info) == 5, 'Found less than 5 outer folds'
    assert all(len(fold_info) == 5 for fold_info in fold_to_ckpt_info.values()), 'Found less than 5 inner folds'

    logging.info(f'Found {len(fold_to_ckpt_info)} nested folds')
    for fold_index_test, fold_info in fold_to_ckpt_info.items():
        logging.info(
            f'Found {len(fold_info)} folds for nested fold {fold_index_test}, '
            f'missing folds: {set(range(5)) - set(fold_info.keys())}'
        )


    n_bootstrap = 1000

    if args.nested:
        fold_indices_test = range(5)
    else:
        fold_indices_test = [0]

    cv_results = defaultdict(dict)
    for fold_index_test in tqdm(fold_indices_test):
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
            data['pac'][fold_index]['y_proba_val'] = y_proba[:, 1]
            data['pac'][fold_index]['gt_val'] = y_val

            # Predict on test
            if args.nested:
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
                _, y_test, _ = get_X_y_groups(cli.datamodule.test_dataset.data)
                data['pac'][fold_index]['y_proba_test'] = y_proba[:, 1]
                data['pac'][fold_index]['gt_test'] = y_test

        if args.nested:
            # Mean test predictions over folds
            data['pac']['mean'] = np.mean(np.stack(list([data['pac'][i]['y_proba_test'] for i in range(5)])), axis=0)

            # GT: same for all folds
            assert np.all([np.all(data['pac'][i]['gt_test'] == data['pac'][0]['gt_test']) for i in range(5)])
            data['gt'] = data['pac'][0]['gt_test']

        if args.nested:
            # Metrics
            # Log loss
            cv_results[fold_index_test]['log_loss'] = log_loss(data['gt'], data['pac']['mean'], eps=1e-16)

            # F1 score
            cv_results[fold_index_test]['f1_score'] = f1_score(
                data['gt'],
                data['pac']['mean'] > 0.5,
            )

            # ROC AUC
            cv_results[fold_index_test]['roc_auc'] = roc_auc_score(
                data['gt'],
                data['pac']['mean'],
            )

            # Bootstrap metrics on n_bootstrap test sets with downsampled negative class
            bootstrap_metrics_dict = bootstrap_metrics(
                data['gt'], 
                data['pac']['mean'],
                n_bootstrap=n_bootstrap,
                replace=True,
            )
            for metric_name, metric_value in bootstrap_metrics_dict.items():
                cv_results[fold_index_test][metric_name] = metric_value

        # Val metrics
        cv_results[fold_index_test]['log_loss_val'] = sum(
            log_loss(data['pac'][i]['gt_val'], data['pac'][i]['y_proba_val'], eps=1e-16)
            for i in range(5)
        ) / 5
        cv_results[fold_index_test]['f1_score_val'] = sum(
            f1_score(data['pac'][i]['gt_val'], data['pac'][i]['y_proba_val'] > 0.5)
            for i in range(5)
        ) / 5
        cv_results[fold_index_test]['roc_auc_val'] = sum(
            roc_auc_score(data['pac'][i]['gt_val'], data['pac'][i]['y_proba_val'])
            for i in range(5)
        ) / 5

        # Bootstrap metrics on n_bootstrap val sets with downsampled negative class
        bootstrap_metrics_dict_outer = defaultdict(list)
        for i in range(5):
            bootstrap_metrics_dict = bootstrap_metrics(
                data['pac'][i]['gt_val'], 
                data['pac'][i]['y_proba_val'],
                n_bootstrap=n_bootstrap,
                replace=True,
            )
            for metric_name, metric_value in bootstrap_metrics_dict.items():
                bootstrap_metrics_dict_outer[metric_name].append(metric_value)
        for metric_name, metric_value in bootstrap_metrics_dict_outer.items():
            cv_results[fold_index_test][metric_name + '_val'] = sum(bootstrap_metrics_dict_outer[metric_name]) / 5

        # Raw data
        cv_results[fold_index_test]['data'] = data

    # Print results
    print_results(cv_results)

    # Save results to file in pickle format
    with open(args.save_path, 'wb') as f:
        pickle.dump(cv_results, f)


if __name__ == '__main__':
    main()
