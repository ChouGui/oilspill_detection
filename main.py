import os
import sys
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# select a gpu
import tensorflow as tf
#print("tf debugging set log device placement True")
#tf.debugging.set_log_device_placement(False)
#print(tf.config.list_physical_devices('GPU'))
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#print("GPUz : ", len(tf.compat.v1.config.experimental.list_physical_devices('GPU')))
import random
# import cv2
import time
# import shutil
# import pickle
#
# import keras
# import tensorflow as tf
#
# from PIL import Image
# from pathlib import Path
# import matplotlib.pyplot as plt
import train
import eval


# The main entry point for this module
def main():
    argu = sys.argv[1:]
    gpu = argu[0]
    context = argu[1]
    launch = argu[2]
    name = None
    if len(argu) > 3:
        name = argu[3]
    # selecting the good gpu
    if context == "bajoo" or context == "cassiopee":
        print("Num GPUs : ", len(tf.compat.v1.config.experimental.list_physical_devices('GPU')))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print("Num GPUs after select : ", len(tf.compat.v1.config.experimental.list_physical_devices('GPU')))
        # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    t1 = time.perf_counter()
    resf = open("result.txt", 'w')
    # Train a model
    if launch == "t" or launch == "tv":
        train.train(resf, context, name)
    t2 = time.perf_counter()
    print(f"train computation time {t2 - t1:0.4f} seconds")
    resf.write(f"model name : {name}\n")
    resf.write(str(f"train computation time {t2 - t1:0.4f} seconds"))
    print("training done, launching evaluation :")
    if launch == "v" or launch == "tv":
        eval.evaluateB(resf, context, name)
        print("evaluation done")
    resf.close()


# Tell python to run main method
if __name__ == '__main__':
    main()
