from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pytorch_lightning.cli import LightningCLI

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")


def cli_main():
    cli = MyLightningCLI()


if __name__ == "__main__":
    cli_main()