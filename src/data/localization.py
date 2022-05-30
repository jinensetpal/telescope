#!/usr/bin/env python3
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

def get_class_activation_map(model, img):
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    label_index = np.argmax(predictions)
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, label_index]

    final_conv_layer = model.get_layer(PENULTIMATE_LAYER) 

    get_output = keras.backend.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    conv_outputs, predictions = get_output([img])
    conv_outputs = np.squeeze(conv_outputs)
    mat_for_mult = sp.ndimage.zoom(conv_outputs, (TARGET_SIZE[0] / conv_outputs.shape[0], TARGET_SIZE[1] / conv_outputs.shape[1], 1), order=1) # dim: 224 x 224 x 2048
    final_output = np.dot(mat_for_mult.reshape((TARGET_SIZE[0] * TARGET_SIZE[1], mat_for_mult.shape[2])), class_weights_winner).reshape(TARGET_SIZE[0], TARGET_SIZE[1]) # dim: 224 x 224

    return final_output, label_index

if __name__ == '__main__':
    from ..const import AUX_SIZE, IMAGE_SIZE, BATCH_SIZE, SEED, N_CHANNELS
    from .generator import Generator
    import imageio

    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'images_variant_train.txt'), sep=' ', header=None, dtype = str)  
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
    model = keras.models.load_model(os.path.join('models', 'localizer-family'))

    fig = plt.figure(figsize=(14, 14),
                    facecolor='white')
    
    for idx in range(BATCH_SIZE):
        out, pred = get_class_activation_map(model, np.resize(test_X[idx], AUX_SIZE + (3,)))
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
    fig.savefig(os.path.join(BASE_DIR, 'data', 'samples', 'cams.png'))
