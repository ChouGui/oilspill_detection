import os
import sys

# Set the different argument from the running command
# 0 : numero of the gpu to use
# 1 : the context of the machine where it is running [home, cass, ping]
# 2 : what to launch [t -> training, v -> validation, tv -> both
# 3 : name of the model
# 4 : nbr of epochs
# 5 : batch size
# 6 : learning rate
# 7 : comment of the run
argu = sys.argv[1:]
gpus = argu[0]
context = argu[1]
launch = argu[2]
epochs = int(argu[4])
batch_size = int(argu[5])
lr = float(argu[6])
comment = argu[7]
name = argu[3] + '.h5'
if context == "ping" or context == "cass":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import warnings

warnings.filterwarnings('ignore')
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
import time
import traceback
import smtplib

# my own folder
import train
import eval
import plotRes


# The main entry point for this module
def main():
    # Mail logins
    mail = 'guillaume.legat@gmail.com'
    passwd = 'wsocfhobvniljwyo'
    tracc = 0
    valacc = 0

    with open('result//error.txt', 'a') as errorf:
        try:
            print(f"RUN : | NAME - {name} | LAUNCH - {launch} | EPOCHS - {epochs} | BATCH SIZE - {batch_size}"
                  f" | LEARNING RATE - {lr} | COMMENTS - {comment}")

            # Train a model
            if launch == "t" or launch == "tv":
                t1 = time.perf_counter()
                histo, tracc, valacc = train.train(gpus, context, name, epochs, batch_size, lr)
                plotRes.plotRes(histo, epochs)
                t2 = time.perf_counter()
                print(f"Trained model name : {name}\n")
                # print(f"tracd code/oi
                # in computation time {t2 - t1:0.2f} seconds")
                print(str(f"train computation time {t2 - t1:0.2f} seconds\n"))
            if launch == "v" or launch == "tv":
                eval.evaluateB(context, name, batch_size)
                # print("evaluation done")

            server = smtplib.SMTP('smtp.googlemail.com', 587)
            server.ehlo()
            server.starttls()
            server.login(mail, passwd)
            BODY = '\r\n'.join(['To: guillaume.legat@gmail.com',
                                'From: guillaume.legat@gmail.com',
                                'Subject: RUN ENDED',
                                '',
                                f"Run ended without crash \n\n\tCONTEXT OF THE RUN : \nNAME \t\t| {name} \nLAUNCH \t| {launch} "
                                f"\nEPOCHS \t| {epochs} \nBATCH SIZE \t| {batch_size}  \nLEARN RATE \t| {lr} "
                                f"\nTRAIN ACC \t| {tracc} \nVAL ACC \t| {valacc}  \nCOMMENTS \t| {comment}"])

            server.sendmail(mail, [mail], BODY)
        except Exception as e:
            server = smtplib.SMTP('smtp.googlemail.com', 587)
            server.ehlo()
            server.starttls()
            server.login(mail, passwd)
            BODY = '\r\n'.join(['To: guillaume.legat@gmail.com',
                                'From: guillaume.legat@gmail.com',
                                'Subject: ERROR OCCURED',
                                '',
                                f"An error occured : {e} \n\n\tCONTEXT OF THE RUN : \nNAME \t\t| {name} \nLAUNCH \t| {launch} "
                                f"\nEPOCHS \t| {epochs} \nBATCH SIZE \t| {batch_size}  \nLEARN RATE \t| {lr} "
                                f"\nTRAIN ACC \t| {tracc} \nVAL ACC \t| {valacc}  \nCOMMENTS \t| {comment}"])
            server.sendmail(mail, [mail], BODY)
            e_type, e_val, e_tb = sys.exc_info()
            traceback.print_exception(e_type, e_val, e_tb, file=errorf)
        server.quit()


# Tell python to run main method
if __name__ == '__main__':
    main()
