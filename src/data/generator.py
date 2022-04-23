#!/usr/bin/env python3
# coding: utf-8

from src.const import BATCH_SIZE, AUX_SIZE, IMAGE_SIZE, BASE_DIR, SEED, N_CLASSES
from src.data.localization import get_class_activation_map, edges, crop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import tensorflow as tf
from glob import glob
from PIL import Image
import numpy as np
import imageio
import random
import os

def create_samples(generator):
    Path('samples').mkdir(parents=True, exist_ok=True)
    for idx, (input_X, input_y) in enumerate(generator):
        imageio.imwrite(os.path.join('samples', f'{idx + 1}_o.png'), input_X['original'][0][:, :, ::-1])
        imageio.imwrite(os.path.join('samples', f'{idx + 1}_c.png'), input_X['cropped'][0][:, :, ::-1])
        imageio.imwrite(os.path.join('samples', f'{idx + 1}_u.png'), input_X['upsampled'][0][:, :, ::-1])

class Generator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size, dim, n_channels, labels, model,
            n_classes, shuffle=True, state="train", augment=None, seed=None):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.labels = labels
        self.shuffle = shuffle
        self.state = state
        self.augment = augment
        self.seed = seed
        self.model = tf.keras.models.load_model(model) if type(model) == str else model

        self.on_epoch_end()
        random.seed(seed)
        self.gen = ImageDataGenerator()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def resolve_path(self, path):
        return path[-2], path[-1].split('.')[0]

    def augmentation_params(self):
        params = {'rescale': 1}

        if 'rescale' in self.augment:
            params['rescale'] = self.augment['rescale']
        if 'horizontal_flip' in self.augment:
            if random.random() > 0.5:
                params['horizontal_flip'] = True
        if 'vertical_flip' in self.augment:
            if random.random() > 0.5:
                params['vertical_flip'] = True
        if 'samplewise_center' in self.augment:
            params['samplewise_center'] = self.augment['samplewise_center']
        if 'samplewise_std_normalization' in self.augment:
            params['samplewise_std_normalization'] = self.augment['samplewise_std_normalization']

        return params 

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization - dim: [TARGET, IMAGE]
        X = {'cropped': np.empty((self.batch_size, *self.dim[0], self.n_channels)),
             'upsampled': np.empty((self.batch_size, *self.dim[0], self.n_channels)),
             'original': np.empty((self.batch_size, *self.dim[1], self.n_channels))}
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            connector, filename = self.resolve_path(ID.split('/'))

            # load images
            X['original'][i,] = Image.open(os.path.join(BASE_DIR, 'data', , 'synthetic', connector, f'{filename}.png'))
            X['cropped'][i,] = X['original'][i,].resize(self.dim[0][::-1])
            y[i] = self.labels[connector]

            cam, pred = get_class_activation_map(self.model, X['cropped'][i,])
            l, r, t, b = edges(cam) + edges(cam.T)

            X['cropped'][i,] = X['cropped'][i,].crop(l, t, r, b).resize(self.dim[0][::-1])
            X['upsampled'][i,] = X['original'][i,].crop(int(l * self.dim[1][0] / self.dim[0][0]), int(r * self.dim[1][0] / self.dim[0][0]), int(t * self.dim[1][1] / self.dim[0][1]), int(b * self.dim[1][1] / self.dim[0][1])).resize(self.dim[0][::-1])

            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X['original'][i,] = self.gen.apply_transform(x=X['original'][i,], transform_parameters=params)
                X['cropped'][i,] = self.gen.apply_transform(x=X['cropped'][i,], transform_parameters=params)
                X['upsampled'][i,] = self.gen.apply_transform(x=X['upsampled'][i,], transform_parameters=params)

        return X, y

if __name__ == '__main__': ## tests every generator
    data_image_paths = glob(os.path.join(BASE_DIR, 'data', , 'synthetic', '*', '*.png'))
    params = {'dim': [AUX_SIZE, IMAGE_SIZE],
            'batch_size': BATCH_SIZE,
            'n_classes': N_CLASSES,
            'n_channels': 3,
            'shuffle': True,
            'model': os.path.join('models', 'cnn-real'),
            'labels': {'4 way MTA100 Plug': 1,
                'AMP M8, Standard Circular Connectors, 4 Position M8 Male': 2,
                'AMP M8, Standard Circular Connectors, 3 Position M8 Male': 3,
                'PCB D-Sub Connectors': 4},
            'augment': {'rescale': 1/255,
                'samplewise_center': True,
                'samplewise_std_normalization': True,
                'horizontal_flip': False,
                'vertical_flip': False}}
    cropped_generator = CroppedGenerator(data_image_paths, state='test', seed=SEED, **params)
    cropped_generator.__getitem__(0)
    create_samples(cropped_generator)
