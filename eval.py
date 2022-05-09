import keras
import os
import pickle
import random
# from PIL import Image
from pathlib import Path


# import matplotlib.pyplot as plt


# Evaluate the model
def evaluateB(resf, context="bajoo", name=None):
    if context == "bajoo" or context == "cassiopee":  # if we are in bajoo config -> big running parameters
        train_path = Path("/linux/glegat/datasets/ann_oil_data/train")
        test_path = Path("/linux/glegat/datasets/ann_oil_data/test ")
        models_path = Path("/linux/glegat/code/oilspill/models")
        epochs = 2
        batch_size = 64
        train_samples = 5000  # 2 categories with 5000 images
        validation_samples = 500  # 10 categories with 1000 images in each category
        steps = None
    else:  # if we are on my computer -> small running parameters
        train_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/train")
        test_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/datasets/ann_oil_data/test")
        models_path = Path("/Users/guillaume/Desktop/UCL/Q100/Memoire/Cassiopee/oilspill/models")
        epochs = 1
        batch_size = 16
        validation_samples = 10  # 10 categories with 1000 images in each category
        steps = 2
    # Load the model
    print(name)
    if name is None:
        model = keras.models.load_model(str(models_path) + "test.h5")
    else:
        model = keras.models.load_model(str(models_path) + '/' + name + '.h5')
    # # Load classes
    # classes = {}
    # with open('models/classes.pkl', 'rb') as file:
    #     classes = pickle.load(file)
    # # Get a list of categories
    # str_test = str(test_path)
    # categories = os.listdir(str_test)
    # # Get a category a random
    # category = random.choice(categories)
    # # Print the category
    # print(category)
    # # Get images in a category
    # images = os.listdir(str_test + '/' + category)
    # # Randomize images to get different images each time
    # random.shuffle(images)

    img_width, img_height = 125, 130
    # Create a data generator for validation
    test_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Create a test generator
    test_generator = test_data_generator.flow_from_directory(
        str(test_path),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=True,
        class_mode='categorical')

    predictions = model.predict(test_generator, batch_size=batch_size, steps=steps)
    resf.write(str(predictions))
    print(predictions)
