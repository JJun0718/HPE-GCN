# -*- coding: utf-8 -*-
# @Time : 2021/9/9 15:46
# @Author : JJun
# @Site :
# @File : confusion_matrix.py.py
# @Software: PyCharm

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_Matrix(y_true, y_pred, classes, savename, title=None):
    cm = confusion_matrix(y_true, y_pred)
    cmap = plt.cm.Blues

    plt.rc('font', family='Times New Roman', size='14')  # 设置字体样式、大小
    plt.figure(figsize=(6, 6.5))

    # Normalize by row
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))

    # Cells that account for less than 1% are set to 0 to prevent them from showing up in the final color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax) # Stripe of color on the side

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           # title=title,
           ylabel='Actual labels',
           xlabel='Predicted labels')

    # Simulate the border of each cell by drawing a grid
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    plt.xlabel('Predicted labels', font2)
    plt.ylabel('Actual labels', font2)

    # plt.setp(ax.get_xticklabels(), ha="right",
    #          rotation_mode="anchor")

    # Labeled percentage information
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, 0,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(savename, dpi=600, format='png')

    # plt.show()