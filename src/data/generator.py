#!/usr/bin/env python3
# coding: utf-8

from ..const import BATCH_SIZE, AUX_SIZE, IMAGE_SIZE, BASE_DIR, SEED, N_CHANNELS, LOCALIZER
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .localization import get_class_activation_map, edges, crop
from cv2 import resize, INTER_CUBIC
from PIL import Image, ImageOps
from pathlib import Path
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import imageio
import random
import os

class Generator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, dim, n_channels, classes,
            shuffle=True, state="train", augment=None, seed=None, localizer=None):
        self.df = df 
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.state = state
        self.augment = augment
        self.seed = seed
        self.localizer = tf.keras.models.load_model(localizer) if type(localizer) == str else localizer
        self.n_classes = len(classes)
        self.classes = {classes[idx]: idx for idx in range(self.n_classes)}

        self.on_epoch_end()
        random.seed(seed)
        self.gen = ImageDataGenerator()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.df[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

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

    @staticmethod
    def crop(img, expected_size):
        img = img.crop((0, 0, img.size[0] - 40, img.size[1] - 60)) # -20 to remove copyright footer, -40 to crop into an optimal aspect ration
        return resize(np.array(img), dsize=expected_size, interpolation=INTER_CUBIC)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples -> X : (n_samples, *dim, n_channels)
        # Initialization - dim: [TARGET, IMAGE]
        X = {'cropped': np.empty((self.batch_size, *self.dim['aux'], self.n_channels)),
             'upsampled': np.empty((self.batch_size, *self.dim['aux'], self.n_channels)),
             'target': np.empty((self.batch_size, *self.dim['aux'], self.n_channels)),
             'original': np.empty((self.batch_size, *self.dim['orig'], self.n_channels))}
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load images
            X['original'][i,] = self.crop(Image.open(os.path.join(BASE_DIR, 'data', 'images', f'{ID[0]}.jpg')).convert('RGB'), self.dim['orig'][::-1])
            X['target'][i,] = resize(X['original'][i,], dsize=self.dim['aux'][::-1], interpolation=INTER_CUBIC)
            y[i] = self.classes[ID[1]]
            if self.localizer:
                cam, pred = get_class_activation_map(self.localizer, X['target'][i,])
                l, r, t, b = edges(cam[0]) + edges(cam[0].T)

                X['cropped'][i,] = np.resize(crop(X['target'][i,], l, r, t, b), self.dim['aux'] + (self.n_channels,))
                X['upsampled'][i,] = np.resize(crop(X['original'][i,], int(l * self.dim['orig'][0] / self.dim['aux'][0]), int(r * self.dim['orig'][0] / self.dim['aux'][0]), int(t * self.dim['orig'][1] / self.dim['aux'][1]), int(b * self.dim['orig'][1] / self.dim['aux'][1])), self.dim['aux'] + (self.n_channels,))

            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X['original'][i,] = self.gen.apply_transform(x=X['original'][i,], transform_parameters=params)
                if self.localizer:
                    X['cropped'][i,] = self.gen.apply_transform(x=X['cropped'][i,], transform_parameters=params)
                    X['upsampled'][i,] = self.gen.apply_transform(x=X['upsampled'][i,], transform_parameters=params)

        if self.state == 'debug':
            return X, y
        if not self.localizer:
            return X['target'], y
        return X['cropped'], y

def create_samples(generator):
    Path(os.path.join(BASE_DIR, 'data', 'samples')).mkdir(parents=True, exist_ok=True)
    input_X, input_y = generator.__getitem__(0)
    for idx in range(BATCH_SIZE): 
        if type(input_X) == dict:
            for datatype in ['original', 'cropped', 'upsampled', 'target']:
                imageio.imwrite(os.path.join(BASE_DIR, 'data', 'samples', 'generator', f'{idx + 1}_{datatype[:3]}.png'), input_X[datatype][idx])
        else: 
            imageio.imwrite(os.path.join(BASE_DIR, 'data', 'samples', 'generator', f'{idx + 1}_orig.png'), input_X[idx])

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'images_variant_train.txt'), sep=' ', header=None, dtype = str)  
    params = {'dim': {'aux': AUX_SIZE, 'orig': IMAGE_SIZE},
            'batch_size': BATCH_SIZE,
            'n_channels': N_CHANNELS,
            'shuffle': True,
            'classes': np.unique(df[1]),
            'augment': {'rescale': 1/255,
                'samplewise_center': True,
                'samplewise_std_normalization': True,
                'horizontal_flip': False,
                'vertical_flip': False}}
    generator = Generator(df.values.tolist(), 
                          state='debug', 
                          # localizer=os.path.join(BASE_DIR, 'models', LOCALIZER),
                          seed=SEED, **params)
    generator.__getitem__(0)
    create_samples(generator)
