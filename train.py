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
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG19

from PIL import Image
from pathlib import Path
#import matplotlib.pyplot as plt


# Train a model
def train(resf, context="bajoo"):
    if context == "bajoo":  # if we are in bajoo config -> big running parameters
        train_path = Path("/linux/glegat/datasets/ann_oil_data/train")
        test_path = Path("/linux/glegat/datasets/ann_oil_data/test")
        models_path = Path("/linux/glegat/code/oilspill/models")
        epochs = 1
        batch_size = 500
        train_samples = 1000  # 2 categories with 5000 images
        validation_samples = 500  # 10 categories with 1000 images in each category
    else:  # if we are on my computer -> small running parameters
        train_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/train")
        test_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test")
        models_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/Sans titre/models")
        epochs = 2
        batch_size = 256
        train_samples = 500  # 2 categories with 5000 images
        validation_samples = 10  # 10 categories with 1000 images in each category
    #f = [x for x in test_path.rglob('*.png')]
    #print(f)
    img_width, img_height = 125, 130
    # Create a data generator for training
    # Making real time data augmentation
    train_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Create a data generator for validation
    validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Create a train generator
    train_generator = train_data_generator.flow_from_directory(
        str(train_path),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=True,
        class_mode='categorical')
    # Create a test generator
    validation_generator = validation_data_generator.flow_from_directory(
        # incorrect ici, je donne le test set en validation -> le test set ne doit être utilisé que après
        str(train_path),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=True,
        class_mode='categorical')
    # print(validation_generator[0])
    # lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5)
    base_model = VGG19(weights=None, input_shape=(125, 130, 1), include_top=False)
    # Weighting the classes because oilspill way less represented
    # model.fit_generator(gen,class_weight=[1.5,0.5]) # gen?
    inputs = keras.Input(shape=(125, 130, 1))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    resf.write(str(model.summary()))

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # in train : not is 4454 and oil : 792 -> class_weight
    model.fit(
        train_generator,
        class_weight={0: 0.15, 1: 0.85},
        # Weighting the classes because oilspill way less represented (0 is not and 1 is oilspill)
        steps_per_epoch=train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        epochs=epochs)

    # Save model to disk
    model_name = 'VGG19_ep' + str(epochs) + '_bs' + str(batch_size) + '_ts' + str(train_samples) + '_vs' + str(
        validation_samples) + '.h5'
    model.save(str(models_path) + model_name)
    print('Saved model to disk!')
    # Get labels
    labels = train_generator.class_indices
    # Invert labels
    classes = {}
    for key, value in labels.items():
        classes[value] = key.capitalize()
    # Save classes to file
    with open('models/classes.pkl', 'wb') as file:
        pickle.dump(classes, file)
    print('Saved classes to disk!')
    return model_name
