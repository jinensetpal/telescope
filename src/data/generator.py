#!/usr/bin/env python3
# coding: utf-8

from src.const import CLASS_MODE, BATCH_SIZE, DATA_TYPE, TARGET_SIZE, IMAGE_SIZE, BASE_DIR, SEED, N_CLASSES, RENDER_SIZE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.visualization.cams import get_class_activation_map
from sklearn.model_selection import train_test_split
from src.data.localization import edges, crop
from pathlib import Path
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import platform
import imageio
import random
import keras
import cv2
import os

def create_samples(generator):
    Path('samples').mkdir(parents=True, exist_ok=True)
    for i, (input_X, input_y) in enumerate(generator):
        imageio.imwrite(os.path.join('samples', f'{i + 1}_o.png'), input_X['original'][0][:, :, ::-1])
        imageio.imwrite(os.path.join('samples', f'{i + 1}_c.png'), input_X['cropped'][0][:, :, ::-1])
        imageio.imwrite(os.path.join('samples', f'{i + 1}_u.png'), input_X['upsampled'][0][:, :, ::-1])


class MultiResGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size, dim, n_channels, downscale_factor,
            n_classes, shuffle=True, state="train", augment=None, seed=None):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.state = state
        self.augment = augment
        self.seed = seed
        self.downscale_factor = downscale_factor
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

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
        # Initialization
        X = np.empty((self.batch_size, *tuple(map(lambda x: x // self.downscale_factor, self.dim)), self.n_channels)) 
        y = np.empty((self.batch_size, *self.dim, self.n_channels)) 

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            connector, filename = self.resolve_path(ID.split('/'))

            # load images
            y[i] = cv2.resize(cv2.imread(os.path.join(BASE_DIR, 'data', 'raw', 'synthetic', connector, f'{filename}.png'), cv2.IMREAD_UNCHANGED), self.dim[::-1])
            X[i,] = cv2.resize(y[i], (y[i].shape[1] // self.downscale_factor, y[i].shape[0] // self.downscale_factor))

            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X[i,] = self.gen.apply_transform(x=X[i,], transform_parameters=params)

        return X, y


class CroppedGenerator(tf.keras.utils.Sequence):
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
        self.model = keras.models.load_model(model) if type(model) == str else model
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
            X['original'][i,] = cv2.imread(os.path.join(BASE_DIR, 'data', DATA_TYPE, 'synthetic', connector, f'{filename}.png'), cv2.IMREAD_UNCHANGED)
            X['cropped'][i,] = cv2.resize(X['original'][i,], self.dim[0][::-1])
            y[i] = self.labels[connector]

            cam, pred = get_class_activation_map(self.model, X['cropped'][i,])
            l, r, t, b = edges(cam) + edges(cam.T)
            print(filename, ':', l, r, t, b)
            X['cropped'][i,] = cv2.resize(crop(X['cropped'][i,], l, r, t, b), self.dim[0][::-1])
            X['upsampled'][i,] = cv2.resize(crop(X['original'][i,], int(l * 1.5), int(r * 1.5), int(t * 2.6667), int(b * 2.6667)), self.dim[0][::-1])

            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X['original'][i,] = self.gen.apply_transform(x=X['original'][i,], transform_parameters=params)
                X['cropped'][i,] = self.gen.apply_transform(x=X['cropped'][i,], transform_parameters=params)
                X['upsampled'][i,] = self.gen.apply_transform(x=X['upsampled'][i,], transform_parameters=params)

        return X, y

def get_generators(BASE_DIR=BASE_DIR,
        data_path=None,
        exclude=None,
        binary=None):
    DELIMITER = r'\\' if platform.system() == 'Windows' else '/'

    data_image_paths = None # for scope
    data_json_paths = None # for scope

    if data_path is None:
        data_image_paths = glob(os.path.join(BASE_DIR, 'data', DATA_TYPE, 'synthetic', '*', '*.png'))
        data_json_paths = glob(os.path.join(BASE_DIR, 'data', DATA_TYPE, 'synthetic', '*', '*.json'))
    else:
        data_image_paths = glob(os.path.join(data_path[0], '*.png'))
        data_json_paths = glob(os.path.join(data_path[1], '*.json'))
    test_data_image_paths = glob(os.path.join(BASE_DIR, 'data', DATA_TYPE, 'photos_640x480', '*', '*', '*', '*.JPG'))

    data_image_paths.sort()
    data_json_paths.sort()
    test_data_image_paths.sort()

    df = pd.read_json(os.path.join(BASE_DIR, 'data', 'processed', 'cumulative_entry.json'))

    df['path'] = data_image_paths
    df['environment'] = list(map(lambda x: x.split(DELIMITER)[-1].split('.')[0], df['environment']))
    df['target'] = list(map(lambda x: x.split(DELIMITER)[-2], data_image_paths))

    if type(binary) is list:
        df['target'] = list(map(lambda x: binary[0] if x == binary[1] else x, df['target']))

    elif binary is not None:
        class_mode = 'binary'
        df['target'] = list(map(lambda x: "1" if x == binary else "0", df['target']))

    real_df = pd.DataFrame(test_data_image_paths, columns=['path'])
    real_df['target'] = list(map(lambda x: x.split(DELIMITER)[-4], real_df['path']))

    if binary is not None:
        real_df['target'] = list(map(lambda x: "1" if x == binary else "0", real_df['target']))

    valid_df, test_df = train_test_split(real_df, test_size=0.20, random_state=2018)

    generator = ImageDataGenerator(rescale=1/255,
            samplewise_center=True,
            samplewise_std_normalization=True,
            horizontal_flip=True,
            vertical_flip=False)

    training_generator = generator.flow_from_dataframe(dataframe=df,
            directory=None,
            x_col='path',
            y_col='target',
            class_mode=CLASS_MODE,
            batch_size=BATCH_SIZE,
            target_size=TARGET_SIZE)

    validation_generator = generator.flow_from_dataframe(dataframe=valid_df,
            directory=None,
            x_col='path',
            y_col='target',
            class_mode=CLASS_MODE,
            batch_size=BATCH_SIZE,
            target_size=TARGET_SIZE)

    test_X, test_y = next(generator.flow_from_dataframe(dataframe=test_df,
        directory=None,
        x_col='path',
        y_col='target',
        class_mode=CLASS_MODE,
        batch_size=256,
        target_size=TARGET_SIZE))

    return training_generator, validation_generator, test_X, test_y

if __name__ == '__main__': ## tests every generator
    get_generators()
    '''
    get_generators(binary='4 way MTA100 Plug')

    data_image_paths = glob(os.path.join(BASE_DIR, 'data', 'processed', 'renders', 'Keyshot', '*', '*.jpg'))
    params = {'dim': RENDER_SIZE, 
            'batch_size': BATCH_SIZE,
            'n_classes': N_CLASSES,
            'n_channels': 3,
            'downscale_factor': 4,
            'shuffle': True,
            'augment': {'rescale': 1/255}}
    multires_generator = MultiResGenerator(data_image_paths, state='train', seed=SEED, **params)
    batch = multires_generator.__getitem__(0)
    create_samples(multires_generator)
    
    data_image_paths = glob(os.path.join(BASE_DIR, 'data', DATA_TYPE, 'synthetic', '*', '*.png'))
    params = {'dim': [TARGET_SIZE, IMAGE_SIZE],
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
    '''
