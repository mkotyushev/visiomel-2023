# 1. Read data folder path from CLI arg
# 2. Print mean of each dimention of shape of images in data folder
# Usage example: 
# python src/scripts/mean_shape.py --input-dir /workspace/data/images_page_7/

import argparse
import numpy as np
import imagesize
import cv2
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='Path to data folder')
parser.add_argument('--in-image-stats', action='store_true', help='If true, calculate in image stats. Takes longer time.')
args = parser.parse_args()

# Read data image by image and save shape to array
filepaths = sorted(list(args.input_dir.glob('**/*.png')))

shapes, colors = [], []
for filepath in tqdm(filepaths):
    w, h = imagesize.get(filepath)
    if args.in_image_stats:
        # Read image and calculate in image stats
        img = cv2.imread(str(filepath))
        colors.append(img.mean(0).mean(0))
    shapes.append((w, h))

# Calcualte mean for each dimention
shapes = np.array(shapes)
print('Mean: ', shapes.mean(0))
print('Min: ', shapes.min(0))
print('Max: ', shapes.max(0))
print('Median: ', np.quantile(shapes, 0.5, 0))

colors = np.array(colors)
print('Mean: ', colors.mean(0))
print('Std: ', colors.std(0))

# Mean:
# 7: [ 836.26304024 1027.62816692]
# 4: [6690.60432191 8221.96870343]
# 7_shrink: [613.59463487 654.74217586]
