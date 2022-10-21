#!/usr/bin/env python3
# coding: utf-8

from tensorflow.keras import models, layers
from keras.applications.vgg16 import VGG16
from .data.generator import Generator
import tensorflow as tf
from . import const
import pandas as pd
import numpy as np
import mlflow
import sys
import os

def get_model(dim, classes, channels=3, primary=True):
    model = models.Sequential()
    vgg = VGG16(include_top=False, pooling='max', input_shape=(*dim, channels))

    for layer in vgg.layers:
        model.add(layer)

    model.add(layers.Flatten())
    if primary:
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model

def get_callbacks():
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()

    return es, reduce_lr

if __name__ == '__main__':
    df = {'train': pd.read_csv(os.path.join(const.BASE_DIR, 'data', const.DATASET_NAME, 'images_family_train.txt'), sep=' ', header=None, dtype=str),
          'validation': pd.read_csv(os.path.join(const.BASE_DIR, 'data', const.DATASET_NAME, 'images_variant_val.txt'), sep=' ', header=None, dtype=str),
          'test': pd.read_csv(os.path.join(const.BASE_DIR, 'data', const.DATASET_NAME, 'images_variant_test.txt'), sep=' ', header=None, dtype=str)}
    params = {'dim': {'aux': const.AUX_SIZE, 'orig': const.IMAGE_SIZE},
            'batch_size': const.BATCH_SIZE,
            'n_channels': const.N_CHANNELS,
            'shuffle': True,
            'classes': np.unique(df['train'][1]),
            'augment': {'rescale': 1/255,
                'samplewise_center': True,
                'samplewise_std_normalization': True,
                'horizontal_flip': False,
                'vertical_flip': False}}
    train = Generator(df['train'].values.tolist(), state='train', seed=const.SEED, **params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-08)

    mlflow.tensorflow.autolog()
    aux = None # extending scope 
    if 'AUX' in sys.argv:
        with mlflow.start_run():
            aux = get_model(const.AUX_SIZE, len(params['classes']), primary=False)
            aux.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            aux.summary()

            aux.fit(train,
                epochs=const.AUX_EPOCHS)
            aux.save(os.path.join(const.BASE_DIR, 'models', 'localizer-family'))
    else:
        aux = tf.keras.models.load_model(os.path.join(const.BASE_DIR, 'models', const.LOCALIZER))

    df['train'] = pd.read_csv(os.path.join(const.BASE_DIR, 'data', const.DATASET_NAME, 'images_variant_train.txt'), sep=' ', header=None, dtype=str) 
    params['localizer'] = aux
    params['classes'] = np.unique(df['train'][1])

    train = Generator(df['train'].values.tolist(), state='train', seed=const.SEED, **params)
    val = Generator(df['validation'].values.tolist(), state='validation', seed=const.SEED, **params)
    test = Generator(df['test'].values.tolist(), state='test', seed=const.SEED, **params)

    with mlflow.start_run():
        classifier = get_model(const.TARGET_SIZE, len(params['classes']))
        classifier.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        aux.summary()
        classifier.summary()

        classifier.fit(train,
                epochs=const.EPOCHS,
                validation_data=val,
                callbacks=get_callbacks())
        classifier.evaluate(test_X, test_y)
        classifier.save(os.path.join(const.BASE_DIR, 'models', 'classifier'))
