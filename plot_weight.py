# -*- coding: utf-8 -*-
# @Time : 2021/10/20 20:41
# @Author : JJun
# @Site : 
# @File : plot_weight.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

TF_IDF = [0.8512, 0.8083, 0.8125, 0.8503, 0.8095, 0.8135]
RANDOM = [0.7625, 0.7500, 0.7476, 0.7619, 0.7619, 0.7528]
CMPE_GCN = [0.8750, 0.8583, 0.8624, 0.8730, 0.8571, 0.8610]

# Set the font style and size
plt.rc('font', family='Times New Roman', size='9')

# Set bar chart color
color1 = "#609dca"
color2 = "#ff963c"
coloe3 = "#38c25d"

# plt.figure(figsize=(6, 6.6))
x = np.arange(6) # The number of groups of data
total_width, n = 0.5, 3
width = total_width / n
x = x - (total_width - width) / 2
real_width = 0.06
plt.bar(x, CMPE_GCN, color=color1, width=real_width, label='FHDD')
plt.bar(x + width, TF_IDF, color=color2, width=real_width, label='TF-IDF')
plt.bar(x + 2 * width,  RANDOM, color=coloe3, width=real_width, label='Random Weight')

# Set the axis font
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
plt.xlabel("Evaluation index", font2)
plt.ylabel("Value", font2)

# Show legend
plt.legend(loc = "best")

# Set coordinate scale
plt.xticks([0, 1, 2, 3, 4, 5],['Macro-P', 'Macro-R', 'Macro-F1', 'Weighted-P', 'Weighted-R', 'Weighted-F1'])
plt.xticks(rotation=45)   # X-axis label rotation

my_y_ticks = np.arange(0.7, 0.89, 0.02)
plt.ylim((0.7, 0.898))  # Actual scale range of y axis
plt.yticks(my_y_ticks)  #  Set the range and interval for displaying Y-axis values


# Show images
plt.tight_layout()  #  Image labels are not blocked
plt.savefig('./Result/weights.png', dpi=600, format='png')
plt.show()