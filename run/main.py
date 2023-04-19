from src.utils.utils import MyLightningCLI, TrainerWandb


def cli_main():
    # TODO: check why configs is overriden when wandb is online
    cli = MyLightningCLI(trainer_class=TrainerWandb, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()