import os
import sys
argu = sys.argv[1:]
context = argu[1]
launch = argu[2]
epochs = int(argu[4])
name = None
if len(argu) > 3:
    name = argu[3] + '.h5'
if context == "bajoo" or context == "cass":
    os.environ["CUDA_VISIBLE_DEVICES"] = argu[0]
import warnings
warnings.filterwarnings('ignore')
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
import random
import time
import traceback
import train
import eval
import plotRes


# The main entry point for this module
def main():
    with open('result//error.txt', 'a') as errorf:
        try:
            resf = open("result/result.txt", 'w')
            os.system("rm done")

            # Train a model
            if launch == "t" or launch == "tv":
                t1 = time.perf_counter()
                _, histo = train.train(resf, context, name, epochs)
                plotRes.plotRes(histo, epochs)
                t2 = time.perf_counter()
                resf.write(f"Trained model name : {name}\n")
                # print(f"train computation time {t2 - t1:0.2f} seconds")
                resf.write(str(f"train computation time {t2 - t1:0.2f} seconds\n"))
            if launch == "v" or launch == "tv":
                eval.evaluateB(resf, context, name)
                # print("evaluation done")
            resf.close()
            os.system("touch done")
            raise ValueError("Some error occurred")
        except Exception as e:
            e_type, e_val, e_tb = sys.exc_info()
            traceback.print_exception(e_type, e_val, e_tb, file=errorf)



# Tell python to run main method
if __name__ == '__main__':
    main()
