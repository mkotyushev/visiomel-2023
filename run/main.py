from src.utils.utils import MyLightningCLI, TrainerWandb


def cli_main():
    cli = MyLightningCLI(trainer_class=TrainerWandb)


if __name__ == "__main__":
    cli_main()