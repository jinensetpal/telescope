#!/usr/bin/env python3
# coding: utf-8

import os

IMAGE_SIZE = (1024, 2048) 
IMAGE_SHAPE = IMAGE_SIZE + (3,)

BASE_DIR = os.getcwd() 
COLOR_MODE = 'rgb'
BATCH_SIZE = 4
PROJECT_NAME = 'telescope'
CLASS_MODE = 'categorical'
EPOCHS = 10
AUX_EPOCHS = 10

SEED = 1024
N_CLASSES = 4
CROP_THRESHOLD = 0.97
PENULTIMATE_LAYER = 'conv2d_4'

N_CHANNELS = 3
DOWNSCALE_FACTOR = 4
MLFLOW_TRACKING = 'https://dagshub.com/jinensetpal/telesecope.mlflow'

TARGET_SIZE = IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4
TARGET_SHAPE = TARGET_SIZE + (3,)
AUX_SIZE = (32, 32)
