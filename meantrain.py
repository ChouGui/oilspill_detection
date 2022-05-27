import os
import sys
import random
import time
import shutil
import pickle

import keras
import tensorflow as tf
from keras.engine.base_layer import Layer
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras import optimizers, Input, Model
from tensorflow.keras.applications import VGG19, VGG16, ResNet50
from tensorflow.python.keras.callbacks import CSVLogger

from pathlib import Path

# PARAMETERS :
models = ['resnet50','vgg19','vgg16']
actis = ['relu','tanh']
d1s = [0, 2048]
d2s = [0, 512]
d3s = [0, 128]

shrs = [0.,0.1,0.2,0.3,0.4,0.5]
zors = [0.,0.1,0.2,0.3,0.4,0.5]
horfls = [True, False]
verfls = [True, False]

bss = [1,2,4,8,16,32]


def cust_model(model="resnet50", d1=2048, d2=512, d3=128, acti='relu'):
    # lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5)
    if model == "resnet50":
        base_model = ResNet50(weights=None, input_shape=(125, 130, 1), include_top=False)
    elif model == "vgg19":
        base_model = VGG19(weights=None, input_shape=(125, 130, 1), include_top=False)
    else:
        base_model = VGG16(weights=None, input_shape=(125, 130, 1), include_top=False)
    inputs = Input(shape=(125, 130, 1))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    if d1 != 0:
        x = Dense(d1, activation=acti)(x)
    if d2 != 0:
        x = Dense(d2, activation=acti)(x)
    if d3 != 0:
        x = Dense(d3, activation=acti)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def get_generator(data_path, bs=16, shuffle=True, shr=0., zor=0., horfl=False, verfl=False):
    img_width, img_height = 125, 130
    # Create a data generator
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=shr,
        zoom_range=zor,
        horizontal_flip=horfl,
        vertical_flip=verfl)
    generator = data_generator.flow_from_directory(
        str(data_path),
        target_size=(img_width, img_height),
        batch_size=bs,
        color_mode='grayscale',
        shuffle=shuffle,
        class_mode='categorical')
    return generator

def get_context(context="cass"):
    if context == "bajoo" or context == "cass" or context == "ping":
        train_path = Path("/linux/glegat/datasets/ann_oil_data/test2")
        test2_path = Path("/linux/glegat/datasets/ann_oil_data/test3")
        models_path = Path("/linux/glegat/code/oilspill_detection/models/")
        train_samples = 516  # 455 + 61 -> 11,82% or 88,18%
        validation_samples = 258  # 228 + 30 -> 11,63% or 88,37%
    else:  # if we are on my computer -> small running parameters
        train_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test2")
        test2_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test3")
        models_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/oilspill/models")
        train_samples = 32  # 2 categories with 5000 images
        validation_samples = 16  # 10 categories with 1000 images in each category
    return train_path, test2_path, models_path, train_samples, validation_samples


def train(model):
    acc = 0
    return acc


def run(gpus="0", context="cass", name=None, epochs=10, batch_size=16, lr=0.0001):
    train_path, test2_path, models_path, train_samples, validation_samples = get_context(context)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Create generators for training and validation
    # Making real time data augmentation
    train_generator = get_generator(str(train_path), batch_size, True)
    validation_generator = get_generator(str(test2_path), batch_size, True)

    os.system("rm result/log.csv")
    csv_logger = CSVLogger('result/log.csv', append=True, separator=';')

    model = cust_model(models[0],d1s[1],d2s[1],d3s[1],actis[0])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(
        train_generator,
        # Weighting the classes because oilspill way less represented (0 is not and 1 is oilspill)
        class_weight={0: 0.12, 1: 0.88},
        steps_per_epoch=train_samples // batch_size,
        batch_size=batch_size,
        # validation_data=validation_generator,
        # validation_steps=validation_samples // batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=[csv_logger])

    tracc = model.evaluate(train_generator,verbose=2,callbacks=[csv_logger])[1]
    valacc = model.evaluate(validation_generator,verbose=2,callbacks=[csv_logger])[1]
    print(tracc)
    print(valacc)
    return history,tracc, valacc
