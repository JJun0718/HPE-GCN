# -*- coding: utf-8 -*-
# @Time : 2021/5/18 9:36
# @Author : JJun
# @Site : 
# @File : standardfile.py
# @Software: PyCharm

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# dataset = 'junheng'
# dataset_file = f'../data/origin/{dataset}3.xlsx'

from config import CONFIG
cfg = CONFIG()
dataset = cfg.dataset

dataset_file = f'../data/origin/buyi_exband.xlsx'

df = pd.read_excel(dataset_file, names=['name', 'labels', 'item'], dtype=str)

df = df[['name', 'labels', 'item']]
# print(df)

index_split = len(df) * 0.9

# print(df[:10])

X = np.arange(len(df.index))
y = df['labels'].values.tolist()

#  X_train: Training set index, X_test: Test set index, y_train: Training set label, y_test: Test set label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

# print(df)
data = df.values.tolist()

#  Real_train: Real training set index, X_test: Validation set index, la_train: Real training set label, la_val: Validation set label
Real_train, Real_val, la_train, la_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)

train_data = []
test_data = []

# train：Real training set + Validation set index
train = np.append(Real_train, Real_val)

for index in train:
    train_data.append(data[index])

for index in X_test:
    test_data.append(data[index])

print("The first 10 formulae：\n", df[:10])
# print(train_data[:5])
# print(test_data[-5:])

train_data.extend([x for x in test_data])
# print(len(train_data))

df = pd.DataFrame(train_data, columns=['name', 'labels', 'item'])

with open(f'../data/corpus/{dataset}.txt', 'w', encoding='utf8') as f:
    for line in df.item:
        f.write(''.join(line)+'\n')

with open(f'../data/corpus/{dataset}.clean.txt', 'w', encoding='utf8') as f:
    for line in df.item:
        f.write(''.join(line)+'\n')


with open(f'../data/{dataset}.txt', 'w', encoding='utf8') as f:
    for index, row in df.iterrows():
        category = 'train' if index < index_split else 'test'
        # single lable
        # f.write(f'{index}\t{category}\t{row[0].split()[sublabel]}\n')
        # multi_label
        name = '\t'.join(row[0].split())
        labels = '\t'.join(row[1].split())
        # num = '\t'.join(row[3].split())
        f.write(f"{name}\t{category}\t{labels}\n")

