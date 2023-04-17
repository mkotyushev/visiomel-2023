from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pytorch_lightning.cli import LightningCLI


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()