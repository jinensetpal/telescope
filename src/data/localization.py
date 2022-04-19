#!/usr/bin/env python
# coding: utf-8

from src.const import BASE_DIR, THRESHOLD
import matplotlib.pyplot as plt
from matplotlib import patches
from tensorflow import keras
import tensorflow as tf
import numpy as np
import imageio
import os

def edges(arr):
    left, right = 9999999, 0
    quant = np.quantile(arr, THRESHOLD)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] > quant: 
                if j < left:
                    left = j 
                if j > right:
                    right = j
    return left, right

def crop(img, l, r, t, b):
    return img[l:r, t:b]

if __name__ == '__main__':
    ## defined under function to avoid circular imports
    from src.visualization.cams import get_class_activation_map
    from src.data.generator import get_generators
    
    train, val, test_X, test_y = get_generators(BASE_DIR)
    model = keras.models.load_model(os.path.join('models', 'cnn-real'))

    fig = plt.figure(figsize=(14, 14),
                    facecolor='white')
    
    for idx in range(100):
            out, pred = get_class_activation_map(model, test_X[idx])

            l, r, t, b = edges(out) + edges(out.T)
            try:
                imageio.imwrite(os.path.join('visualizations', 'localization', 'original', f'{idx}.png'), test_X[idx]) 
                imageio.imwrite(os.path.join('visualizations', 'localization', 'cropped', f'{idx}.png'), crop(test_X[idx], l, r, t, b))
            except:
                print('failed:', idx)

    for idx in range(16):
        out, pred = get_class_activation_map(model, test_X[idx])
        l, r, t, b = edges(out) + edges(out.T)

        fig.add_subplot(4, 4, idx + 1)
        buf = f'Bounds: L: {l} R: {r}, T: {t}, B: {b}'
        plt.xlabel(buf)
        plt.imshow(test_X[idx], alpha=0.5)
        plt.imshow(out, cmap='jet', alpha=0.5)
        plt.gca().add_patch(patches.Rectangle(
            (l, t), r - l, b - t,
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'))

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join('visualizations', 'cams-crops.png'))
