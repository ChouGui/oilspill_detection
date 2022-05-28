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
import smtplib


def create_model():
    # lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5)
    # base_model = VGG16(weights=None, input_shape=(125, 130, 1), include_top=False)
    base_model = ResNet50(weights=None, input_shape=(125, 130, 1), include_top=False)
    inputs = Input(shape=(125, 130, 1))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def get_generator(data_path, bs=16, shuffle=True):
    img_width, img_height = 125, 130
    # Create a data generator
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
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
        train_path = Path("/linux/glegat/datasets/ann_oil_data/train")
        test2_path = Path("/linux/glegat/datasets/ann_oil_data/test2")
        models_path = Path("/linux/glegat/code/oilspill_detection/models/")
        train_samples = 5246  # 4454 + 792
        validation_samples = 516  # 455 + 61
    else:  # if we are on my computer -> small running parameters
        train_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/train")
        test2_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test2")
        models_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/oilspill/models")
        train_samples = 32  # 2 categories with 5000 images
        validation_samples = 16  # 10 categories with 1000 images in each category
    return train_path, test2_path, models_path, train_samples, validation_samples


# Train a model
def train(gpus="0", context="cass", name=None, epochs=10, batch_size=16, lr=0.0001):
    train_path, test2_path, models_path, train_samples, validation_samples = get_context(context)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Create generators for training and validation
    # Making real time data augmentation
    train_generator = get_generator(str(train_path), batch_size, True)
    validation_generator = get_generator(str(test2_path), batch_size, True)

    os.system("rm result/log.csv")
    csv_logger = CSVLogger('result/log.csv', append=False, separator=';')

    model = create_model()
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(
        train_generator,
        # Weighting the classes because oilspill way less represented (0 is not and 1 is oilspill)
        class_weight={0: 0.15, 1: 0.85},
        steps_per_epoch=train_samples // batch_size,
        batch_size=batch_size,
        # validation_data=validation_generator,
        # validation_steps=validation_samples // batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=[csv_logger])
    # callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))])

    print("##### EVALUATION #####")
    tracc = model.evaluate(train_generator, steps=train_samples // batch_size, batch_size=batch_size, verbose=2,
                   callbacks=[csv_logger])[1]
    valacc = model.evaluate(validation_generator, steps=validation_samples // batch_size, batch_size=batch_size, verbose=2,
                   callbacks=[csv_logger])[1]
    print("Train prediction : ")
    print(model.predict(train_generator, batch_size=batch_size, steps=train_samples // batch_size))
    print("Valid prediction : ")
    print(model.predict(validation_generator, batch_size=batch_size, steps=validation_samples // batch_size))

    model.save(str(models_path) + '/' + name)
    # Get labels
    labels = train_generator.class_indices
    # Invert labels
    classes = {}
    for key, value in labels.items():
        classes[value] = key.capitalize()
    # Save classes to file
    with open('models/classes.pkl', 'wb') as file:
        pickle.dump(classes, file)
    return history, tracc, valacc

    # convert array into dataframe
    # DF = pd.DataFrame(arr)
    # save the dataframe as a csv file
    # DF.to_csv("data1.csv")
