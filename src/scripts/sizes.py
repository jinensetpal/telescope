#!/usr/bin/env python3
# coding: utf-8

from src.const import BASE_DIR
from PIL import Image
import pandas as pd
import os

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'images_variant_train.txt'), sep=' ', header=None, dtype = str)
    width, height = [], []
    for ID in df[0]:
        img = Image.open(os.path.join(BASE_DIR, 'data', 'images', f'{ID}.jpg')).convert('RGB')
        width.append(img.size[0])
        height.append(img.size[1])
    df['height'], df['width'] = height, width
    print(df.head())
    df.to_csv(os.path.join(BASE_DIR, 'data', 'images_variant_train.txt'), sep=' ', index=False, header=False)
