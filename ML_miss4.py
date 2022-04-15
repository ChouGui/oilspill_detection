'''
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import norm
from pandas import *
from scipy import stats
'''
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger

((x_train, y_train), (x_valid, y_valid)) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train)
y_valid = keras.utils.to_categorical(y_valid)


def create_model_431_best_regu():
    # cks:3,cp:same,cki:RandomNormal,cbi:RandomNormal, mp:same,dki:GlorotNormal,dbi:RandomNormal,dkr:l1,dbr:l2,acc:0.5838
    model = keras.Sequential()
    model.add(keras.Input(shape=(32,32,3)))

    model.add(keras.layers.Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='RandomNormal', bias_initializer='RandomNormal'))
    model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
    model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='RandomNormal', bias_initializer='RandomNormal'))
    model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
    model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='RandomNormal', bias_initializer='RandomNormal'))
    model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
    model.add(keras.layers.Flatten())

    #model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='GlorotNormal', bias_initializer='RandomNormal',kernel_regularizer = keras.regularizers.l2(1e-4), bias_regularizer = keras.regularizers.l1(1e-4)))
    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='GlorotNormal', bias_initializer='RandomNormal',kernel_regularizer = 'l2', bias_regularizer = keras.regularizers.l1_l2()))

    model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer='RandomNormal', bias_initializer='RandomNormal'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy',metrics='accuracy')

    return model

def launch(resf):
    # save the best found for now
    model_name = 'cks:3,cp:same,cki:RandomNormal,cbi:RandomNormal, mp:same,dki:GlorotNormal,dbi:RandomNormal,dkr:l2,dbr:l1_l2'
    name = 'fitted432d.model'
    csv_logger = CSVLogger('log.csv', append=True, separator=';')

    model1 = create_model_431_best_regu()
    model1.fit(x_train, y_train, batch_size=128, epochs=300, verbose=2, callbacks=[csv_logger])
    a = round(model1.evaluate(x_valid, y_valid)[1], 4)
    model1.save('models/ep300' + name, save_format='h5')
    resf.write("model de type " + model_name + "\n")
    resf.write(f"{name} 300 epochs acc : {a}\n")

    model1 = create_model_431_best_regu()
    model1.fit(x_train, y_train, batch_size=128, epochs=250, verbose=2, callbacks=[csv_logger])
    a = round(model1.evaluate(x_valid, y_valid)[1], 4)
    model1.save('models/ep250' + name, save_format='h5')
    resf.write("model de type " + model_name + "\n")
    resf.write(f"{name} 250 epochs acc : {a}\n")
    # fitted432ep150.model acc : 0.6616

resf = open("result.txt", 'w')
launch(resf)
resf.close()