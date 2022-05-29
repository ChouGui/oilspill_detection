import os
import smtplib
import sys
import traceback

context = sys.argv[1]
if context == "ping" or context == "cass":
    gpus = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
m = sys.argv[3]

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import tensorflow as tf

import utils
import params

# The main entry point for this module
def main():
    train_path, test2_path, models_path, tsam, vsam = utils.get_context(context)
    tgen = utils.get_generator(str(train_path), params.bs, shuffle=True, shr=0.2, zor=0.2, horfl=True, verfl=False)
    model = utils.cust_model(params.models[int(m)],d1=0, d2=512, d3=128, acti='relu')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy',
                  metrics=['acc'])
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.01, patience=3, min_lr=0.001)

    history = model.fit(
        tgen,
        # Weighting the classes because oilspill way less represented (0 is not and 1 is oilspill)
        class_weight={0: 0.12, 1: 0.88},
        steps_per_epoch=tsam // params.bs,
        batch_size=params.bs,
        # validation_data=vgen,
        # validation_steps=vsam // batch_size,
        epochs=params.epochs,
        verbose=2)  # ,
    # callbacks=[reduce_lr])
    ac = history.history['acc'][params.epochs - 1]
    acc = round(ac, 3)
    print(f"acc : {acc}")


# Tell python to run main method
if __name__ == '__main__':
    main()