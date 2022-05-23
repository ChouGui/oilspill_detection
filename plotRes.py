import os
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_csv():
    a = 0
    #path = Path("/Users/guillaume/Desktop/")

def plotRes(histo, ep):
    rcParams['figure.figsize'] = (18, 8)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    fig = plt.figure()
    plt.plot(
        np.arange(1, ep+1),
        histo.history['loss'],
        label='Loss', lw=3
    )
    plt.plot(
        np.arange(1, ep+1),
        histo.history['acc'],
        label='Accuracy', lw=3
    )
    #plt.plot(np.arange(1, ep+1),histo.history['lr'],label='Learning rate', color='pink', lw=3, linestyle='--')
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.legend()
    fig.savefig('result/resplot.jpg')#, bbox_inches='tight', dpi=150)
    #plt.show()
