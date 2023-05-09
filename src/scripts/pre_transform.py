# 1. Read data folder path from CLI arg
# 2. Read output data folder path from CLI arg
# 3. Read output data folder path downscale factor from CLI arg
# 3. For each image in data folder downscale it by downscale factor and save it to output data folder
# Usage example: 
# python src/scripts/pre_transform.py --input-dir /workspace/data/images_page_4/ --output-dir /workspace/data/images_page_4_shink/ --scale 8

import argparse
from pathlib import Path
from PIL import Image
import pyvips
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from src.data.transforms import CenterCropPct, Shrink
from torchvision.transforms import Compose, Resize

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='Path to data folder')
parser.add_argument('--output-dir', type=Path, help='Path to output folder')
parser.add_argument('--scale', type=int, help='Preview scale for shrink transform')
parser.add_argument('--img-type', type=str, default='png', choices=['png', 'tif'], help='Resize to that size (squared) as last transform')
parser.add_argument('--img-size', type=int, default=None, help='Resize to that size (squared) as last transform')
args = parser.parse_args()

img_mean = (238, 231, 234)
if args.img_size is None:
    pre_transform = Compose(
        [
            CenterCropPct(size=(0.9, 0.9)),
            Shrink(scale=args.scale, fill=img_mean),
        ]
    )
else:
    pre_transform = Compose(
        [
            CenterCropPct(size=(0.9, 0.9)),
            Shrink(scale=args.scale, fill=img_mean),
            Resize(size=(args.img_size, args.img_size)),
        ]
    )

extension = 'png' if args.img_type == 'png' else 'tif'
scale_to_tif_page = {
    8: 4,
    1: 7,
}

# Read data image by image, pre-transform it and save it to output directory
filepaths = sorted(list(args.input_dir.glob(f'**/*.{extension}')))
for filepath in tqdm(filepaths):
    if args.img_type == 'png':
        image = Image.open(str(filepath))
    else:
        image = pyvips.Image.new_from_file(str(filepath), page=scale_to_tif_page[args.scale])
        # image = image.convert('RGB')
        # image = Image.fromarray(image.to_memoryview().cast('B').tobytes(), 'RGB')
        image = Image.fromarray(image.numpy())
    image = pre_transform(image)
    
    # Create a dir structure if it doesn't exist
    output_filepath = args.output_dir / filepath.relative_to(args.input_dir)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_filepath))
