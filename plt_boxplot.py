# -*- coding: utf-8 -*-
# @Time : 2021/12/19 22:24
# @Author : JJun
# @Site : 
# @File : plt_boxplot.py
# @Software: PyCharm

import pandas as pd
from pylab import *
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['Times New Roman']

data = {
    'TF-IDF':        [0.6818, 0.6364, 0.6364, 0.6818, 0.6818, 0.6818, 0.7273, 0.7727, 0.7727, 0.6818],
    'Random Weight': [0.6818, 0.8182, 0.6364, 0.7727, 0.7273, 0.6364, 0.6818, 0.7727, 0.8182, 0.6364],
    'FHDD':          [0.8636, 0.8636, 0.7273, 0.8636, 0.8182, 0.8182, 0.7727, 0.8182, 0.8636, 0.7727],
}
df = pd.DataFrame(data)

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}



# df.plot.box(title="Consumer spending in each country", vert=False)
df.plot(kind='box', fontsize = 'large')

plt.ylabel("Accuracy", font2)  # Y label
plt.grid(linestyle="--", alpha=0.3)
plt.savefig('../Result/箱线图.png', dpi=600, format='png')
plt.show()