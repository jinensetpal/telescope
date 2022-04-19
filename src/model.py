#/usr/bin/env python3
# coding: utf-8

from tensorflow.keras import models, layers
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np
import os

def get_model(dim, classes, channels=3, primary=True):
    model = models.Sequential()
    vgg = VGG16(include_top=False, pooling='max', input_shape=(*dim, channels))

    for layer in vgg.layers:
        model.add(layer)

    model.add(layers.Flatten())
    if primary:
        model.add(layers.Dense(1024, activation='relu')
        model.add(layers.Dropout(0.4)
        model.add(layers.Dense(1024, activation='relu')
        model.add(layers.Dense(256, activation='relu')

    model.add(layers.Dense(64, activation='relu')
    model.add(layers.Dense(16, activation='relu')
    model.add(layers.Dense(classes, activation='softmax'))
    return model

def get_callbacks():
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()

    return es, reduce_lr

if __name__ == '__main__':
    params = {'dim': const.IMAGE_SIZE,
            'batch_size': const.BATCH_SIZE,
            'n_classes': const.N_CLASSES,
            'n_channels': 3,
            'shuffle': True,
            'localizer': os.path.join('models', 'localizer'),
            'labels': {'4 way MTA100 Plug': 1,
                'AMP M8, Standard Circular Connectors, 4 Position M8 Male': 2,
                'AMP M8, Standard Circular Connectors, 3 Position M8 Male': 3,
                'PCB D-Sub Connectors': 4},
            'augment': {'rescale': 1/255,
                'samplewise_center': True,
                'samplewise_std_normalization': True,
                'horizontal_flip': False,
                'vertical_flip': False}}
    train = CroppedGenerator(train_image_paths, state='train', seed=SEED, aux=True, **params)
    aux = auxilliary_model(const.TARGET_SIZE, const.N_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-08)
    aux.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    aux.fit(train,
        epochs=const.AUX_EPOCHS)
    model.save(BASE_DIR, 'models', 'localizer')
    params['localizer'] = aux

    train = CroppedGenerator(train_image_paths, state='train', seed=SEED, **params)
    valid = CroppedGenerator(valid_image_paths, state='valid', seed=SEED, **params)
    test = CroppedGenerator(test_image_paths, state='test', seed=SEED, **params)

    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        classifier = primary_model(const.N_CLASSES)
        classifier.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        aux.summary()
        classifier.summary()

        classifier.fit(train,
                epochs=const.EPOCHS,
                validation_data=val,
                callbacks=get_callbacks())
        classifier.evaluate(test_X, test_y)
        model.save(os.path.join(BASE_DIR, 'models', 'classifier'))
