# 1. Read data folder path from CLI arg
# 2. Read output data folder path from CLI arg
# 3. Read output data folder path downscale factor from CLI arg
# 3. For each image in data folder downscale it by downscale factor and save it to output data folder
# Usage example: 
# python src/utils/downscale.py --input-dir /workspace/data/images_page_7/ --output-dir /workspace/data/images_page_6/ --downscale-factor 2

import argparse
import gc
import os
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='Path to data folder')
parser.add_argument('--output-dir', type=Path, help='Path to output folder')
parser.add_argument('--downscale-factor', type=int, help='Downscale factor')
args = parser.parse_args()

# Read data image by image, downscale it and save it to output directory
filepaths = sorted(list(args.input_dir.glob('**/*.png')))
for filepath in tqdm(filepaths):
    image = cv2.imread(str(filepath))
    image = cv2.resize(
        image, 
        (
            image.shape[1] // args.downscale_factor, 
            image.shape[0] // args.downscale_factor
        ), 
        interpolation = cv2.INTER_AREA
    )
    output_filepath = args.output_dir / filepath.relative_to(args.input_dir)
    
    # Create a dir structure if it doesn't exist
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_filepath), image)
