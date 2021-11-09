# -*- coding: utf-8 -*-
# @Time : 2021/9/14 16:16
# @Author : JJun
# @Site :
# @File : calculate.py
# @Software: PyCharm

from plot_confusion_matrix import plot_Matrix

y_true =        [2,1,0,2,2,1,0,0,3,1,3,3,1,0,3,2,2,1,3,0,3]

bpnn_y_pred =   [2,1,0,2,2,2,0,0,3,0,3,0,1,0,2,2,2,0,3,3,3]
cnn_y_pred =    [2,1,0,3,2,2,0,0,3,1,3,0,1,0,3,2,2,0,3,3,3]
rnn_y_pred =    [2,1,0,2,2,2,0,0,3,0,3,0,1,0,3,2,2,0,3,3,3]
lstm_y_pred =   [2,1,0,2,2,2,0,0,3,1,3,0,1,0,3,2,2,0,3,3,3]
gcn_y_pred =    [2,1,0,2,2,1,0,0,3,1,3,0,1,0,3,2,2,0,3,3,3]



classes = ['0', '1', '2', '3']

plot_Matrix(y_true, bpnn_y_pred, classes, 'Result/BPNN_confusion_matrix.png')
plot_Matrix(y_true, cnn_y_pred, classes, 'Result/CNN_confusion_matrix.png')
plot_Matrix(y_true, rnn_y_pred, classes, 'Result/RNN_confusion_matrix.png')
plot_Matrix(y_true, lstm_y_pred, classes, 'Result/LSTM_confusion_matrix.png')
plot_Matrix(y_true, gcn_y_pred, classes, 'Result/CMPE_GCNn_confusion_matrix.png')