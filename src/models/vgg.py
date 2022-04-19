#/usr/bin/env python
# coding: utf-8

from src.data.generator import rs_generators
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from keras.applications.vgg16 import VGG16
from pathlib import Path
import tensorflow as tf
import numpy as np
import keras
import PIL
import sys
import os

def build_model():
    model = models.Sequential()
    vgg = VGG16(include_top=False, classes=3, pooling='max', input_shape=(320,240,3))
    for layer in vgg.layers:
        model.add(layer)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()
    return model

if __name__ == '__main__':
    synthetic_path = os.path.join(BASE_DIR, 'synthetic_3way')
    real_path = os.path.join(BASE_DIR, 'real_3way')
    sys.path.append('..')
    train, val, test = get_generators(BASE_DIR,
            synthetic=synthetic_path,
            real=real_path)

    model = build_model()

    checkpoint_path = "Checkpoints/vgg.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model compiled...")

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
    for i in range(1, 21):
        if (i != 1):
            model.load_weights(checkpoint_path)
        print("Epoch #", i)
        model.fit(train, epochs=1, validation_data=val, callbacks=[cp_callback, reduce_lr], verbose=2) 
        model.evaluate(test)
    print("Model trained...")

    model.evaluate(test)
    model.save("models/vgg")
