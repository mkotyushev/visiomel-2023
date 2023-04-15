import sys
from pytorch_lightning.cli import LightningCLI


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    # Predict from final.ckpt by default
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                'predict', 
                '--config', 'run/configs/fake_config.yaml', 
                '--ckpt_path', 'final.ckpt'
            ]
        )
    cli_main()