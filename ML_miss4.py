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

((x_train, y_train), (x_valid, y_valid)) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train)
y_valid = keras.utils.to_categorical(y_valid)

# Create a conv model
# You are free to play with the meta-parameters of each of these layers, as long as you respect this structure.
# the best should be like
#bp = bp.append({'cks':4,'cp':'same','cdr':1,'cki':'RN','cbi':'RN', 'mp':'valid','dki':None,'dbi':None,'acc':0.6369}, ignore_index=True)
def create_model_431(ck = 3, cp = 'same', cki = 'GlorotNormal', cbi = 'RandomNormal', mp = 'same', dki = 'GlorotNormal',dbi = 'RandomNormal', dkr = None, dbr = None):
    if dkr == 'l1_l2':
      dkr = keras.regularizers.l1_l2()
    elif dkr == 'l1_4':
      dkr = keras.regularizers.l1(1e-4)
    elif dkr == 'l2_4':
      dkr = keras.regularizers.l2(1e-4)
    if dbr == 'l1_l2':
      dbr = keras.regularizers.l1_l2()
    elif dbr == 'l1_4':
      dbr = keras.regularizers.l1(1e-4)
    elif dbr == 'l2_4':
      dbr = keras.regularizers.l2(1e-4)
    model = keras.Sequential()
    model.add(keras.Input(shape=(32,32,3)))
    # 3 conv blocks nbr filters 16, 32, 64 and each containing
      # Conv2D with ReLU
      # Max Pooling layer (2,2) size
    model.add(keras.layers.Conv2D(16, ck,padding = cp, activation='relu', kernel_initializer=cki, bias_initializer=cbi))
    model.add(keras.layers.MaxPooling2D((2,2),padding = mp))
    model.add(keras.layers.Conv2D(32, ck,padding = cp, activation='relu', kernel_initializer=cki, bias_initializer=cbi))
    model.add(keras.layers.MaxPooling2D((2,2),padding = mp))
    model.add(keras.layers.Conv2D(64, ck,padding = cp, activation='relu', kernel_initializer=cki, bias_initializer=cbi))
    model.add(keras.layers.MaxPooling2D((2,2),padding = mp))

    model.add(keras.layers.Flatten())
    # A dense layer with 32 hidden units
    model.add(keras.layers.Dense(32,activation = 'relu', kernel_initializer=dki, bias_initializer=dbi, kernel_regularizer = dkr, bias_regularizer = dbr))
    # A dense output layer with a softmax activation
    model.add(keras.layers.Dense(10,activation = 'softmax', kernel_initializer='RandomNormal', bias_initializer='RandomNormal'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy',metrics='accuracy')

    return model

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

    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='GlorotNormal', bias_initializer='RandomNormal',kernel_regularizer = keras.regularizers.l2(1e-4), bias_regularizer = keras.regularizers.l1(1e-4)))

    model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer='RandomNormal', bias_initializer='RandomNormal'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy',metrics='accuracy')

    return model

def launch(resf):
    # save the best found for now
    model_name = 'cks:3,cp:same,cki:RandomNormal,cbi:RandomNormal, mp:same,dki:GlorotNormal,dbi:RandomNormal,dkr:l2_4,dbr:l1_4'
    name = 'fitted432ep200b.model'
    model1 = create_model_431_best_regu()
    model1.fit(x_train, y_train, batch_size=128, epochs=200)
    a = round(model1.evaluate(x_valid, y_valid)[1], 4)
    model1.save('models/' + name, save_format='h5')
    resf.write("model de type "+model_name+"\n")
    resf.write("accu avec 50 epochs : ?\n")
    resf.write(f"{name} 200 epochs acc : {a}\n")
    # fitted432ep150.model acc : 0.6616

resf = open("result.txt", 'w')
launch(resf)
resf.close()