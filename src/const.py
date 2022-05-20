#!/usr/bin/env python3
# coding: utf-8

import os

# smallest dimensions - (775, 413)
IMAGE_SIZE = (1600, 1205) # largest dimensions 
IMAGE_SHAPE = IMAGE_SIZE + (3,)

BASE_DIR = os.getcwd() 
COLOR_MODE = 'rgb'
BATCH_SIZE = 2 
PROJECT_NAME = 'telescope'
CLASS_MODE = 'categorical'
EPOCHS = 5
AUX_EPOCHS = 2

SEED = 1024
THRESHOLD = 0.97
PENULTIMATE_LAYER = 'block5_conv3'

N_CHANNELS = 3
DOWNSCALE_FACTOR = 4
MLFLOW_TRACKING = 'https://dagshub.com/jinensetpal/telesecope.mlflow'

TARGET_SIZE = IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4
TARGET_SHAPE = TARGET_SIZE + (3,)
AUX_SIZE = IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16
