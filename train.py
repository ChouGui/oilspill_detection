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

# from PIL import Image
from pathlib import Path


# import matplotlib.pyplot as plt


def create_model():
    # lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5)
    base_model = VGG16(weights=None, input_shape=(125, 130, 1), include_top=False)
    #base_model = ResNet50(weights=None, input_shape=(125, 130, 1), include_top=False)
    # Weighting the classes because oilspill way less represented
    # model.fit_generator(gen,class_weight=[1.5,0.5]) # gen?
    inputs = Input(shape=(125, 130, 1))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(128, activation='tanh')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


# Train a model
def train(resf, context="cass", name=None, epochs=10):
    if context == "bajoo" or context == "cass":  # if we are in bajoo config -> big running parameters
        train_path = Path("/linux/glegat/datasets/ann_oil_data/train")
        test_path = Path("/linux/glegat/datasets/ann_oil_data/test")
        test2_path = Path("/linux/glegat/datasets/ann_oil_data/test2")
        models_path = Path("/linux/glegat/code/oilspill_detection/models/")
        #epochs = 100
        batch_size = 32
        train_samples = 5246  # 2 categories with 5000 images
        validation_samples = 516  # 10 categories with 1000 images in each category
    else:  # if we are on my computer -> small running parameters
        train_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/train")
        test_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test")
        test2_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test2")
        models_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/oilspill/models")
        #epochs = 2
        batch_size = 16
        train_samples = 32  # 2 categories with 5000 images
        validation_samples = 16  # 10 categories with 1000 images in each category
    # f = [x for x in test_path.rglob('*.png')]
    # print(f)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    img_width, img_height = 125, 130
    # Create a data generator for training
    # Making real time data augmentation
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Create a data generator for validation
    validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
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
        str(test2_path),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=True,
        class_mode='categorical')
    # print(validation_generator[0])

    csv_logger = CSVLogger('result/log.csv', append=False, separator=';')

    model = create_model()

    #model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy',metrics=['acc'])
    print(train_generator)
    #print(train_generator[0])
    a = train_generator[0] # taille 2
    print(len(a[0])) # 16
    print(len(a[1])) # 16
    b = a[0]
    print(len(b[0]))
    print(b[0])
    print(len(b[0][0][0]))

    #for i in train_generator:

        #print(len(i))
    # in train : not is 4454 and oil : 792 -> class_weight
    history = model.fit(
        train_generator,
        class_weight={0: 0.15, 1: 0.85},
        # Weighting the classes because oilspill way less represented (0 is not and 1 is oilspill)
        steps_per_epoch=train_samples // batch_size,
        #validation_data=validation_generator,
        #validation_steps=validation_samples // batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=[csv_logger])
        # callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))])
    print(history)

    tr = model.evaluate(train_generator, steps=train_samples // batch_size, batch_size=batch_size, verbose=2, callbacks=[csv_logger])
    va = model.evaluate(validation_generator, steps=validation_samples // batch_size, batch_size=batch_size, verbose=2, callbacks=[csv_logger])
    print(f"train loss - acc : {tr}")
    print(f"valid loss - acc : {va}")
    predtr = model.predict(train_generator, batch_size=batch_size, steps=train_samples // batch_size)
    # convert array into dataframe
    #DF = pd.DataFrame(arr)
    # save the dataframe as a csv file
    #DF.to_csv("data1.csv")
    predva = model.predict(validation_generator, batch_size=batch_size, steps=validation_samples // batch_size)
    predf = open("result/predict.txt", 'w')
    predf.write("PRINT PREDICTION OF TRAIN :\n")
    for p in predtr:
        predf.write(str(p)+"\n")
    predf.write("PRINT PREDICTION OF VALIDATION :\n")
    for p in predva:
        predf.write(str(p)+"\n")
    predf.close()
    print(predtr)
    print(predva)

    # Save model to disk
    if name is None:
        name = 'VGG19_ep' + str(epochs) + '_bs' + str(batch_size) + '_ts' + str(train_samples) + '_vs' + str(
            validation_samples) + '.h5'
    model.save(str(models_path) + '/' + name)
    #print('Saved model to disk!')
    # Get labels
    labels = train_generator.class_indices
    # Invert labels
    classes = {}
    for key, value in labels.items():
        classes[value] = key.capitalize()
    # Save classes to file
    with open('models/classes.pkl', 'wb') as file:
        pickle.dump(classes, file)
    #print('Saved classes to disk!')
    return name,history
