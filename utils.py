
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras import optimizers, Input, Model
from tensorflow.keras.applications import VGG19, VGG16, ResNet50, DenseNet121
from tensorflow.python.keras.callbacks import CSVLogger

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams





# UTILS
def cust_model(model="resnet50", d1=2048, d2=512, d3=128, acti='relu'):
    # lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5)
    if model == "resnet50":
        base_model = ResNet50(weights=None, input_shape=(125, 130, 1), include_top=False)
    elif model == "vgg19":
        base_model = VGG19(weights=None, input_shape=(125, 130, 1), include_top=False)
    #elif model == "resnet101":
        #base_model = ResNet101(weights=None, input_shape=(125, 130, 1), include_top=False)
    elif model == "densenet121":
        base_model = DenseNet121(weights=None, input_shape=(125, 130, 1), include_top=False)
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


def get_generator(data_path, bs=16, shuffle=True, shr=0.2, zor=0.2, horfl=True, verfl=False):
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

def get_context(context="drive"):
    if context == "bajoo" or context == "cass" or context == "ping":
        train_path = Path("/linux/glegat/datasets/ann_oil_data/test2")
        test2_path = Path("/linux/glegat/datasets/ann_oil_data/test3")
        models_path = Path("/linux/glegat/code/oilspill_detection/models/")
        train_samples = 516  # 455 + 61 -> 11,82% or 88,18%
        validation_samples = 258  # 228 + 30 -> 11,63% or 88,37%
    elif context == "drive":
        train_path = Path("/content/drive/MyDrive/Colab_Notebooks/Memoire/crop_DB/test2")
        test2_path = Path("/content/drive/MyDrive/Colab_Notebooks/Memoire/crop_DB/test3")
        models_path = Path("/content/drive/MyDrive/Colab_Notebooks/Memoire/crop_DB/models/")
        train_samples = 516  # 455 + 61 -> 11,82% or 88,18%
        validation_samples = 292 #258  # 228 + 30 -> 11,63% or 88,37%
    else:  # if we are on my computer -> small running parameters
        train_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test2")
        test2_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test3")
        models_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/oilspill/models")
        train_samples = 32  # 2 categories with 5000 images
        validation_samples = 16  # 10 categories with 1000 images in each category
    return train_path, test2_path, models_path, train_samples, validation_samples

# PLOT MEAN
def plotmean(x,means,name = "mean0acc",title = 'Evaluation of epochs',xlabel = 'Epoch'):
    labels = ['VGG16','VGG19','ResNet50','ResNet101','DenseNet121']
    rcParams['figure.figsize'] = (18, 8)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    fig = plt.figure()
    plt.title(title, size=20)
    plt.xlabel(xlabel, size=14)
    plt.ylabel('Accuracy')
    plt.ylim(top=1)
    plt.ylim(bottom=0)
    for i in range(len(means)):
      plt.plot(
          x,
          means[i],
          label=labels[i], lw=3
      )
    plt.legend()
    fig.savefig(f"mean/{name}.jpg")
    plt.show()