# 1. Read .csv labels from CLI arg using argparse
# 2. Read data folder path from CLI arg
# 3. Create a subfolder in data folder for each class in metadata 'relapse' column
# 4. Move all images from data folder to the corresponding class subfolder
# Usage example: 
# python src/scripts/split_by_classes.py --data /workspace/data/images_page_7/ --labels /workspace/data/train_labels.csv

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to data folder')
parser.add_argument('--labels', type=str, help='Path to labels .csv file')
args = parser.parse_args()

# Read labels .csv file
df = pd.read_csv(args.labels)

# Create a subfolder for each class in data folder
for i in df['relapse'].unique():
    os.makedirs(os.path.join(args.data, str(i)))

# Move all images from data folder to the corresponding class subfolder
for i in range(len(df)):
    filename = df.iloc[i]['filename'].split('.')[0] + '.png'
    filepath = os.path.join(args.data, filename)
    # Check if image exists in data folder
    if not os.path.exists(filepath):
        continue
    os.rename(
        filepath, 
        os.path.join(args.data, str(df.iloc[i]['relapse']), filename)
    )
