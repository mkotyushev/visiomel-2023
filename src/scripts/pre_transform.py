# 1. Read data folder path from CLI arg
# 2. Read output data folder path from CLI arg
# 3. Read output data folder path downscale factor from CLI arg
# 3. For each image in data folder downscale it by downscale factor and save it to output data folder
# Usage example: 
# python src/utils/downscale.py --input-dir /workspace/data/images_page_7/ --output-dir /workspace/data/images_page_6/ --downscale-factor 2

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.data.transforms import CenterCropPct, Shrink
from torchvision.transforms import Compose, Resize

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='Path to data folder')
parser.add_argument('--output-dir', type=Path, help='Path to output folder')
args = parser.parse_args()

img_size = 512
pre_transform = Compose(
    [
        CenterCropPct(size=(0.9, 0.9)),
        Shrink(scale=None),
        Resize(size=(img_size, img_size)),
    ]
)

# Read data image by image, pre-transform it and save it to output directory
filepaths = sorted(list(args.input_dir.glob('**/*.png')))
for filepath in tqdm(filepaths):
    image = Image.open(str(filepath))
    image = pre_transform(image)
    
    # Create a dir structure if it doesn't exist
    output_filepath = args.output_dir / filepath.relative_to(args.input_dir)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_filepath))
