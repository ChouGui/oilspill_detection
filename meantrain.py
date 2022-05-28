import os
import smtplib
import sys
import traceback

context = sys.argv[1]
if context == "ping" or context == "cass":
    gpus = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import pandas as pd
import tensorflow as tf

import utils
import params


def run(xs, context="cass", epochs=10, batch_size=16, lr=0.0001):
    train_path, test2_path, models_path, tsam, vsam = utils.get_context(context)
    totruns = len(xs) * len(params.models) * params.nbrruns
    # Create generators for training and validation
    # Making real time data augmentation
    # v6, v9, r50, r101, d121 = [], [], [], [], []
    valmeans = [[], [], [], [], []]
    accmeans = [[], [], [], [], []]
    a = 0
    for x in xs:  # XS IS LR

        tgen = utils.get_generator(str(train_path), batch_size)
        vgen = utils.get_generator(str(test2_path), batch_size)

        for m in range(len(params.models)):
            accmean = 0
            valmean = 0
            for i in range(params.nbrruns):
                print(f"({a}/{totruns}) : {x} - {params.models[m]} - [{i}/{params.nbrruns}]")
                a += 1
                model = utils.cust_model(params.models[m])
                # model.summary()
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=x), loss='categorical_crossentropy',
                              metrics=['acc'])
                # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.01, patience=3, min_lr=0.001)

                history = model.fit(
                    tgen,
                    # Weighting the classes because oilspill way less represented (0 is not and 1 is oilspill)
                    class_weight={0: 0.12, 1: 0.88},
                    steps_per_epoch=tsam // batch_size,
                    batch_size=batch_size,
                    validation_data=vgen,
                    validation_steps=vsam // batch_size,
                    epochs=epochs,
                    verbose=0)  # ,
                # callbacks=[reduce_lr])

                acc = round(history.history['acc'][epochs - 1], 3)
                valacc = round(history.history['val_acc'][epochs - 1], 3)
                print(f"acc : {acc} - valacc : {valacc}")
                valmean += valacc
                accmean += acc
            valmeans[m].append(round(valmean / params.nbrruns, 3))
            accmeans[m].append(round(accmean / params.nbrruns, 3))
        print(f"{x} just done")
        print("accmeans")
        print(accmeans)
        print("valmeans")
        print(valmeans)
    # convert array into dataframe
    accdf = pd.DataFrame(accmeans)
    valdf = pd.dataFrame(valmeans)
    # save the dataframe as a csv file
    accdf.to_csv("mean/accmean.csv")
    valdf.to_csv("mean/valmean.csv")
    return accmeans, valmeans

# The main entry point for this module
def main():
    with open('result//error.txt', 'a') as errorf:
        try:
            print(f"RUN : | NAME - {params.name} | EPOCHS - {params.epochs} | BATCH SIZE - {params.bs}"
                  f" | LEARNING RATE - {params.lr} | COMMENTS - {params.comment}")
            accmeans,valmeans = run(params.xs, context, params.epochs, params.bs, params.lr)
            #plotmean(x,means,title = 'Evaluation of epochs',xlabel = 'Epoch')
            utils.plotmean(params.xs,accmeans,params.name+"acc","Training Accuracy", params.xlabel)
            utils.plotmean(params.xs,valmeans,params.name+"val","Validation Accuracy", params.xlabel)

            # send a mail when finish
            server = smtplib.SMTP('smtp.googlemail.com', 587)
            server.ehlo()
            server.starttls()
            server.login(params.mail, params.passwd)
            BODY = '\r\n'.join(['To: guillaume.legat@gmail.com',
                                'From: guillaume.legat@gmail.com',
                                'Subject: RUN FINISHED',
                                '',
                                f"Run ended without crash \n\n\tCONTEXT OF THE RUN : \nNAME \t\t| {params.name} \n"
                                f"\nEPOCHS \t| {params.epochs} \nBATCH SIZE \t| {params.bs}  \nLEARN RATE \t| {params.lr} "
                                f"\nCOMMENTS \t| {params.comment}\n XS \t\t| {params.xs}"])
            server.sendmail(params.mail, [params.mail], BODY)
            server.quit()
        except Exception as e:
            server = smtplib.SMTP('smtp.googlemail.com', 587)
            server.ehlo()
            server.starttls()
            server.login(params.mail, params.passwd)
            BODY = '\r\n'.join(['To: guillaume.legat@gmail.com',
                                'From: guillaume.legat@gmail.com',
                                'Subject: ERROR OCCURED',
                                '',
                                f"An error occured : {e} \n\n\tCONTEXT OF THE RUN : \nNAME \t\t| {params.name} \n"
                                f"\nEPOCHS \t| {params.epochs} \nBATCH SIZE \t| {params.bs}  \nLEARN RATE \t| {params.lr} "
                                f"\nCOMMENTS \t| {params.comment}\n XS \t\t| {params.xs}"])
            server.sendmail(params.mail, [params.mail], BODY)
            server.quit()
            e_type, e_val, e_tb = sys.exc_info()
            traceback.print_exception(e_type, e_val, e_tb, file=errorf)

# Tell python to run main method
if __name__ == '__main__':
    main()