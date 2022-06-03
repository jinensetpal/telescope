#!/usr/bin/env python2
# coding: utf-8

from ..const import BASE_DIR, THRESHOLD, PENULTIMATE_LAYER, TARGET_SIZE
import matplotlib.pyplot as plt
from matplotlib import patches
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sp
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

def get_class_activation_map(model, images):
    label_index, final_outputs = [], []
    if len(images.shape) < 4:
        images = np.expand_dims(images, axis=0)

    for idx, prediction in enumerate(model.predict(images)):
        label_index.append(np.argmax(prediction))
        class_weights = model.layers[-2].get_weights()[0]
        class_weights_winner = class_weights[:, label_index]

        final_conv_layer = model.get_layer(PENULTIMATE_LAYER) 

        get_output = keras.backend.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        conv_outputs, predictions = get_output([np.expand_dims(images[idx], axis=0)])
        conv_outputs = np.squeeze(conv_outputs)
        mat_for_mult = sp.ndimage.zoom(conv_outputs, (TARGET_SIZE[0] / conv_outputs.shape[0], TARGET_SIZE[1] / conv_outputs.shape[1], 1), order=1) # dim: 400 x 301 x 512
        final_outputs.append(np.dot(mat_for_mult.reshape((TARGET_SIZE[0] * TARGET_SIZE[1], class_weights_winner.shape[0])), class_weights_winner).reshape(TARGET_SIZE[0], TARGET_SIZE[1])) # dim: 400 x 301

    return final_outputs, label_index

if __name__ == '__main__':
    from ..const import AUX_SIZE, IMAGE_SIZE, BATCH_SIZE, SEED, N_CHANNELS, LOCALIZER
    from cv2 import resize, INTER_CUBIC
    from .generator import Generator
    from PIL import Image

    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'images_variant_train.txt'), sep=' ', header=None, dtype=str)
    params = {'dim': [AUX_SIZE, IMAGE_SIZE],
            'batch_size': BATCH_SIZE,
            'n_channels': N_CHANNELS,
            'shuffle': True,
            'classes': np.unique(df[1]),
            'augment': {'rescale': 1/255,
                'samplewise_center': True,
                'samplewise_std_normalization': True,
                'horizontal_flip': False,
                'vertical_flip': False}}
    generator = Generator(df.values.tolist(), state='train', seed=SEED, **params)
    test_X, test_y = generator.__getitem__(0)
    model = keras.models.load_model(os.path.join('models', LOCALIZER))

    fig = plt.figure(figsize=(14, 14),
                    facecolor='white')
    
    for idx in range(BATCH_SIZE):
        img = resize(test_X[idx], dsize=AUX_SIZE[::-1], interpolation=INTER_CUBIC)
        out, pred = get_class_activation_map(model, img)
        img = resize(test_X[idx], dsize=TARGET_SIZE[::-1], interpolation=INTER_CUBIC)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        l, r, t, b = edges(out[0]) + edges(out[0].T)

        fig.add_subplot(4, 4, idx + 1)
        buf = f'Bounds: L: {l} R: {r}, T: {t}, B: {b}'
        plt.xlabel(buf)
        plt.imshow(img, alpha=0.5)
        plt.imshow(out[0], cmap='jet', alpha=0.5)
        plt.gca().add_patch(patches.Rectangle(
            (l, t), r - l, b - t,
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'))

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(BASE_DIR, 'data', 'samples', 'cams', f'{LOCALIZER.split("-")[1]}.png'))
