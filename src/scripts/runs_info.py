# For each dir (checkpoint_dir) in --checkpoints-dir (each dirname in checkpoints_dir has format '{name}'), 
# find wandb dir (run_dir_name) in --wandb-logs-dir (each dirname in wandb_logs_dir has format 'run-{time}-{name}')
# and extract data.init_args.fold_index from run_dir_name / files / config_pl.yaml

# Also get checkpoint_dir / checkpoints / *.ckpt (single file) path

import argparse
from pathlib import Path
import yaml


def get_runs_info(checkpoints_dir: Path, wandb_logs_dir: Path):
    wandb_logs_subdirs = [
        d for d in 
        wandb_logs_dir.iterdir()
        if d.is_dir() and (
            d.name.startswith('run-') or
            d.name.startswith('offline-run-')
        )
    ]

    run_info = {}
    for checkpoint_dir in checkpoints_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        name = checkpoint_dir.name
        for wandb_dir in wandb_logs_subdirs:
            if name not in wandb_dir.name:
                continue
            config_path = wandb_dir / 'files' / 'config_pl.yaml'
            if not config_path.exists():
                print(f'no config.yaml for {name} in {config_path}, skipping')
                continue
            with config_path.open() as f:
                config = yaml.safe_load(f)
            if 'fold_index_test' not in config['data']['init_args']:
                print(f'no fold_index_test in config.yaml for {name} in {config_path}, skipping')
                continue
            run_info[name] = {
                'fold_index_test': config['data']['init_args']['fold_index_test'],
                'fold_index': config['data']['init_args']['fold_index'],
                'checkpoint_paths': list(map(str, (checkpoint_dir / 'checkpoints').glob('*.ckpt'))),
                'config_path': config_path,
            }            

    return run_info 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-dir', type=Path, required=True)
    parser.add_argument('--wandb-logs-dir', type=Path, required=True)
    args = parser.parse_args()

    run_info = get_runs_info(args.checkpoints_dir, args.wandb_logs_dir)
    print(run_info)


if __name__ == '__main__':
    main()