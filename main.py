import os
import sys
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


# The main entry point for this module
def main():
    t1 = time.perf_counter()
    resf = open("result.txt", 'w')
    # Train a model
    model_name = train.train(resf, sys.argv[1])
    t2 = time.perf_counter()
    resf.write(f"model name : {model_name}")
    resf.write(str(f"train computation time {t2 - t1:0.4f} seconds"))
    resf.close()


# Tell python to run main method
if __name__ == '__main__':
    main()
