import tensorflow as tf
import keras
import os
import pickle
import random
# from PIL import Image
from pathlib import Path

import train


# import matplotlib.pyplot as plt


# Evaluate the model
def evaluateB(context="cass", name="test", batch_size=16):
    train_path, test2_path, models_path, train_samples, validation_samples = train.get_context(context)
    # Load the model
    model = tf.keras.models.load_model(str(models_path) + '/' + name)

    # Create generators for training and test
    # Making real time data augmentation
    train_generator = train.get_generator(str(train_path), batch_size, True)
    test_generator = train.get_generator(str(test2_path), batch_size, False)

    predtr = model.predict(train_generator, batch_size=batch_size, steps=train_samples // batch_size)
    predtest = model.predict(test_generator, batch_size=batch_size, steps=validation_samples // batch_size)
    predf = open("result/predict.txt", 'w')
    predf.write("\n")
    predf.write("#########################################\n")
    predf.write("############## EVALUTATION ##############\n")
    predf.write("#########################################\n")
    predf.write("\n")
    testclass = test_generator.classes
    predf.write(str(len(predtest)) + " - " + str(len(testclass)) + "\n")
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(predtest)):
        predf.write("val) " + str(predtest[i]) + " - " + str(testclass[i]) + "\n")
        if int(testclass[i]) and predtest[i][1] >= predtest[i][0]:  # TP : classe == 1 and pred[1] >= prd[0]
            TP += 1
        elif int(testclass[i]) and predtest[i][1] < predtest[i][0]:  # FP : classe == 1 and pred[1] < prd[0]
            FP += 1
        elif int(testclass[i]) == 0 and predtest[i][1] <= predtest[i][0]:  # TN : classe == 0 and pred[1] <= prd[0]
            TN += 1
        elif int(testclass[i]) == 0 and predtest[i][1] > predtest[i][0]:  # FN : classe == 0 and pred[1] > prd[0]
            FN += 1
    predf.write(f"\nON EVAL WITH {len(testclass)} examples : {TP} TP, {FP} FP, {TN} TN and {FN} FN\n")
    print(f"\nON EVAL WITH {len(testclass)} examples : {TP} TP, {FP} FP, {TN} TN and {FN} FN")

    predf.write("\n##########################################\n")
    predf.write("################ TRAINING ################\n")
    predf.write("##########################################\n")
    trclass = train_generator.classes
    predf.write(str(len(predtr)) + " - " + str(len(trclass)) + "\n")
    for i in range(len(predtr)):
        predf.write("tr) " + str(predtr[i]) + " - " + str(trclass[i]) + "\n")
    predf.close()


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
