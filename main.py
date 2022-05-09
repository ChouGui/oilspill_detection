import os
import sys
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# select a gpu
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
    t1 = time.perf_counter()
    # resf = open("result.txt", 'w')
    # # Train a model
    # context = sys.argv[1]
    # model_name = train.train(resf, context)
    t2 = time.perf_counter()
    print(f"train computation time {t2 - t1:0.4f} seconds")
    # resf.write(f"model name : {model_name}")
    # resf.write(str(f"train computation time {t2 - t1:0.4f} seconds"))
    # print("training done, launching evaluation :")
    # eval.evaluateB(resf,context, model_name)
    # print("evaluation done")
    # resf.close()


# Tell python to run main method
if __name__ == '__main__':
    main()
