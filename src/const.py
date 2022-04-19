#!/usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path

IMAGE_SIZE = (480, 640) 
IMAGE_SHAPE = IMAGE_SIZE + (3,)
TARGET_SIZE = (320, 240)
TARGET_SHAPE = TARGET_SIZE + (3,)
COLOR_MODE = 'rgb'
BATCH_SIZE = 4
PROJECT_NAME = 'connector-classification'

# Dataset params
CLASS_MODE = 'categorical'
EPOCHS = 20

BASE_DIR = Path(os.getcwd()) #.resolve().parents[0]
PROD_MODEL_PATH = os.path.join(BASE_DIR,'models')

DATA_TYPE = 'raw' #  displays the type of the data that is in use -- after data processing is complete, this will change accordingly
THRESHOLD = 0.97
SEED = 1024
N_CLASSES = 4
PENULTIMATE_LAYER = 'conv2d_4'

MLFLOW_TRACKING = 'https://dagshub.com/jinensetpal/connector-classification.mlflow'
DOWNSCALE_FACTOR = 4
N_CHANNELS = 3

RENDER_SIZE = IMAGE_SIZE 
RENDER_SHAPE = RENDER_SIZE + (3,)
DOWNSCALED_SIZE = (RENDER_SIZE[0] // DOWNSCALE_FACTOR, RENDER_SIZE[1] // DOWNSCALE_FACTOR)
DOWNSCALED_SHAPE = DOWNSCALED_SIZE + (3,)
