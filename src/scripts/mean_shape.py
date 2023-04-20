# 1. Read data folder path from CLI arg
# 2. Print mean of each dimention of shape of images in data folder
# Usage example: 
# python src/utils/mean_shape.py --input-dir /workspace/data/images_page_7/

import argparse
import numpy as np
import imagesize
import re
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='Path to data folder')
args = parser.parse_args()

# Read data image by image and save shape to array
filepaths = sorted(list(args.input_dir.glob('**/*.png')))

shapes = []
for filepath in tqdm(filepaths):
    w, h = imagesize.get(filepath)
    shapes.append((w, h))

# Calcualte mean for each dimention
shapes = np.array(shapes)
print(shapes.mean(0))

# 7: [ 836.26304024 1027.62816692]
# 4: [6690.60432191 8221.96870343]
# 7_shrink: [613.59463487 654.74217586]
