# -*- coding: utf-8 -*-
# @Time : 2021/9/14 16:16
# @Author : JJun
# @Site :
# @File : calculate.py
# @Software: PyCharm

from plot_confusion_matrix import plot_Matrix

y_true =      [3, 1, 0, 3, 0, 1, 3, 3, 2, 1, 0, 2, 1, 3, 2, 3, 0, 2, 1, 2, 0, 1]

svm_y_pred =  [3, 1, 0, 3, 0, 1, 3, 3, 2, 2, 0, 2, 1, 3, 2, 3, 2, 2, 1, 1, 0, 1]
nb_y_pred =   [0, 3, 0, 3, 0, 1, 2, 3, 2, 0, 0, 2, 1, 3, 2, 0, 2, 2, 1, 1, 0, 1]
lr_y_pred =   [3, 1, 0, 3, 0, 1, 3, 3, 2, 2, 0, 2, 1, 3, 2, 3, 2, 2, 1, 1, 0, 1]
gbdt_y_pred = [3, 3, 0, 3, 0, 1, 3, 3, 2, 2, 0, 2, 1, 3, 2, 3, 1, 2, 1, 1, 0, 3]
knn_y_pred =  [3, 1, 0, 3, 0, 1, 3, 3, 2, 2, 0, 2, 1, 3, 2, 3, 1, 2, 1, 1, 0, 1]
gcn_y_pred =  [3, 1, 0, 3, 0, 1, 3, 3, 2, 2, 0, 2, 1, 3, 2, 3, 0, 2, 1, 1, 0, 1]



classes = ['0', '1', '2', '3']

plot_Matrix(y_true, svm_y_pred, classes, 'Result/confusion_matrix/svm_confusion_matrix.png')
plot_Matrix(y_true, nb_y_pred, classes, 'Result/confusion_matrix/nb_confusion_matrix.png')
plot_Matrix(y_true, lr_y_pred, classes, 'Result/confusion_matrix/lr_confusion_matrix.png')
plot_Matrix(y_true, gbdt_y_pred, classes, 'Result/confusion_matrix/gbdt_confusion_matrix.png')
plot_Matrix(y_true, knn_y_pred, classes, 'Result/confusion_matrix/knn_confusion_matrix.png')
plot_Matrix(y_true, gcn_y_pred, classes, 'Result/confusion_matrix/HPE_GCN_confusion_matrix.png')
