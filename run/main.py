from src.utils.utils import MyLightningCLI, TrainerWandb


def cli_main():
    cli = MyLightningCLI(
        trainer_class=TrainerWandb, 
        save_config_kwargs={
            'config_filename': 'config_pl.yaml',
            'overwrite': True,
        }
    )


if __name__ == "__main__":
    cli_main()