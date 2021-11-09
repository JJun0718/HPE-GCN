# -*- coding: utf-8 -*-
# @Time : 2021/9/8 10:41
# @Author : JJun
# @Site :
# @File : build_FHHG.py
# @Software: PyCharm

import os
import random
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
from math import log
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, mahalanobis

import sys
sys.path.append('../')
from utils.utils import loadHerb2Vec

from config import CONFIG
cfg = CONFIG()
dataset = cfg.dataset

# Read herb Vectors
# herb_vector_file = '../data/corpus/' + dataset + '_herb_vectors.txt'
# _, embd, herb_vector_map = loadHerb2Vec(herb_vector_file)
# herb_embeddings_dim = len(embd[0])

# herb_embeddings_dim = 300
# herb_vector_map = {}

formula_name_list = []
formula_train_list = []
formula_test_list = []

with open('../data/' + dataset + '.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        formula_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            formula_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            formula_train_list.append(line.strip())
# print(formula_train_list)
# print(formula_test_list)

formula_content_list = []
with open('../data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        formula_content_list.append(line.strip())
# print(formula_content_list)

train_ids = []
for train_name in formula_train_list:
    train_id = formula_name_list.index(train_name)
    train_ids.append(train_id)
# print(train_ids)
random.shuffle(train_ids)

# partial labeled data

train_ids_str = '\n'.join(str(index) for index in train_ids)
with open('../data/' + dataset + '.train.index', 'w', encoding='utf8') as f:
    f.write(train_ids_str)

test_ids = []
for test_name in formula_test_list:
    test_id = formula_name_list.index(test_name)
    test_ids.append(test_id)
# print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
with open('../data/' + dataset + '.test.index', 'w', encoding='utf8') as f:
    f.write(test_ids_str)

ids = train_ids + test_ids

shuffle_formula_name_list = []
shuffle_formula_herbs_list = []
for id in ids:
    shuffle_formula_name_list.append(formula_name_list[int(id)])
    shuffle_formula_herbs_list.append(formula_content_list[int(id)])
shuffle_formula_name_str = '\n'.join(shuffle_formula_name_list)
shuffle_formula_herbs_str = '\n'.join(shuffle_formula_herbs_list)

with open('../data/' + dataset + '_shuffle.txt', 'w', encoding='utf8') as f:
    f.write(shuffle_formula_name_str)

with open('../data/corpus/' + dataset + '_shuffle.txt', 'w', encoding='utf8') as f:
    f.write(shuffle_formula_herbs_str)

# build vocab
herb_freq = {}  #  frequency of herb occurrence
herb_set = set()   #  all herb names
herb_all_len = {}    #  The number of all herbs contained in the formula in which the herb is present

for formula_herbs in shuffle_formula_herbs_list:
    herbs = formula_herbs.split()
    for herb in herbs:
        herb_set.add(herb)
        if herb in herb_freq:
            herb_freq[herb] += 1
            herb_all_len[herb] += len(herbs)
        else:
            herb_freq[herb] = 1
            herb_all_len[herb] = len(herbs)

vocab = list(herb_set)
vocab_size = len(vocab)   # number of herbs

herb_formula_list = {}

for i in range(len(shuffle_formula_herbs_list)):
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    appeared = set()
    for herb in herbs:
        if herb in appeared:
            continue
        if herb in herb_formula_list:
            formula_list = herb_formula_list[herb]
            formula_list.append(i)
            herb_formula_list[herb] = formula_list
        else:
            herb_formula_list[herb] = [i]
        appeared.add(herb)

herb_formula_freq = {}
for herb, formula_list in herb_formula_list.items():
    herb_formula_freq[herb] = len(formula_list)

# the average length of the formula in which a herb appears
herb_avg_len = {}
for herb in vocab:
    herb_avg_len[herb] = herb_all_len[herb]/herb_formula_freq[herb]
# ----

herb_id_map = {}
for i in range(vocab_size):
    herb_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

with open('../data/corpus/' + dataset + '_vocab.txt', 'w', encoding='utf8') as f:
    f.write(vocab_str)

'''
herb definitions begin
'''

# dataFrame = pd.read_excel('../data/origin/zhongyaodatabase.xlsx')
# df = dataFrame[['中药名', '寒', '热', '温', '凉', '平', '酸', '苦', '甘', '辛', '咸', '肺', '心包', '心', '大肠', '三焦', '小肠', '胃', '胆', '膀胱', '脾', '肝', '肾', '毒性']]
#
# name = df['中药名'].values.tolist()
# print(name)

# print(vocab)

# df = df.values.tolist()
# herb_vector_dict = {}
#
# for item in df:
#     for i in range(1, len(item)):
#         if type(item[i]) == float:
#             item[i] = 0.
#         else:
#             item[i] = 1.
#     herb_vector_dict[item[0]] = item[1:]

xinwei_xls = pd.read_excel("../data/origin/zhongyaodatabase.xlsx")  # reading TCM-HPs data
name = xinwei_xls["中药名"].values.tolist()  # Get herb name
herb_vector_dict = {}  # Dictionary type, key: herb name, value: TCM-HPs
for index, med in enumerate(xinwei_xls.values):  # Circulation of quantitative
    med_name = med[0]
    med_xinwei = list(range(23))
    med_xinwei[0] = 2 if med[1] == "大寒" else 0                  # severely cold
    med_xinwei[0] = 1 if med[1] == "寒" else med_xinwei[0]        # cold
    med_xinwei[0] = 0.5 if med[1] == "微寒" else med_xinwei[0]    # slightly cold
    med_xinwei[1] = 2 if med[2] == "大热" else 0                  # severely hot
    med_xinwei[1] = 1 if med[2] == "热" else med_xinwei[1]        # hot
    med_xinwei[1] = 0.5 if med[2] == "微热" else med_xinwei[1]    # slightly hot
    med_xinwei[2] = 1 if med[3] == "温" else 0                    # warm
    med_xinwei[2] = 0.5 if med[3] == "微温" else med_xinwei[2]    # slightly warm
    med_xinwei[3] = 1 if med[4] == "凉" else 0                    # cool
    med_xinwei[3] = 0.5 if med[4] == "微凉" else med_xinwei[3]     # slightly cool
    med_xinwei[4] = 1 if med[5] == "平" else 0                    # neutral
    med_xinwei[5] = 1 if med[6] == "酸" else 0                    # sour
    med_xinwei[5] = 0.5 if med[6] == "微酸" else med_xinwei[5]    # slightly sour
    med_xinwei[6] = 1 if med[7] == "苦" else 0                    # bitter
    med_xinwei[6] = 0.5 if med[7] == "微苦" else med_xinwei[6]    # slightly bitter
    med_xinwei[7] = 1 if med[8] == "甘" else 0                    # sweet
    med_xinwei[7] = 0.5 if med[8] == "微甘" else med_xinwei[7]     # slightly sweet
    med_xinwei[8] = 1 if med[9] == "辛" else 0                    # pungent
    med_xinwei[8] = 0.5 if med[9] == "微辛" else med_xinwei[8]     # slightly pungent
    med_xinwei[9] = 1 if med[10] == "咸" else 0                    # salty
    med_xinwei[9] = 0.5 if med[10] == "微咸" else med_xinwei[9]    # slightly salty
    med_xinwei[10] = 1 if med[11] == "肺" else 0                   # lung
    med_xinwei[11] = 1 if med[12] == "心包" else 0                 # pericardium
    med_xinwei[12] = 1 if med[13] == "心" else 0                   # heart
    med_xinwei[13] = 1 if med[14] == "大肠" else 0                 # large intestine
    med_xinwei[14] = 1 if med[15] == "三焦" else 0                 # triple energizer
    med_xinwei[15] = 1 if med[16] == "小肠" else 0                 # small intestine
    med_xinwei[16] = 1 if med[17] == "胃" else 0                   # stomach
    med_xinwei[17] = 1 if med[18] == "胆" else 0                   # gallbladder
    med_xinwei[18] = 1 if med[19] == "膀胱" else 0                 # bladder
    med_xinwei[19] = 1 if med[20] == "脾" else 0                   # spleen
    med_xinwei[20] = 1 if med[21] == "肝" else 0                   # liver
    med_xinwei[21] = 1 if med[22] == "肾" else 0                   # kidney
    med_xinwei[22] = 2 if med[23] == "大毒" else 0                  # severely toxic
    med_xinwei[22] = 1 if med[23] == "有毒" else med_xinwei[22]     # having toxic
    med_xinwei[22] = 0.5 if med[23] == "小毒" else med_xinwei[22]   # slightly toxic

    herb_vector_dict[med_name] = med_xinwei  # Deposit each herb in "med_xw_dic", indexed by the drug name

ret = [i for i in vocab if i not in name]
if len(ret) != 0:
    with open(f'../data/origin/{dataset}herbs in dispute.txt', 'w', encoding='utf8') as f:
        num = 0
        for herb in ret:
            f.write(herb + ' ')
            num = num + 1
            if num % 10 == 0:
                f.write('\n')
    f.close()

herb_vectors = []

key_list = list(herb_vector_dict.keys())
for i in range(len(vocab)):
    herb = vocab[i]
    if herb in key_list:
        vector = herb_vector_dict[herb]
        str_vector = []
        for j in range(len(vector)):
            str_vector.append(str(vector[j]))
        temp = ' '.join(str_vector)
        herb_vector = herb + ' ' + temp
        herb_vectors.append(herb_vector)

string = '\n'.join(herb_vectors)

f = open('../data/corpus/' + dataset + '_herb_vectors.txt', 'w', encoding='utf8')
f.write(string)
f.close()

herb_vector_file = '../data/corpus/' + dataset + '_herb_vectors.txt'
_, embd, herb_vector_map = loadHerb2Vec(herb_vector_file)
herb_embeddings_dim = len(embd[0])


# label list
label_set = set()
for formula_meta in shuffle_formula_name_list:
    temp = formula_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
with open('../data/corpus/' + dataset + '_labels.txt', 'w', encoding='utf8') as f:
    f.write(label_list_str)

# x: feature vectors of training formulas, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_formula_names = shuffle_formula_name_list[:real_train_size]
real_train_formula_names_str = '\n'.join(real_train_formula_names)

with open('../data/' + dataset + '.real_train.name', 'w', encoding='utf8') as f:
    f.write(real_train_formula_names_str)

print("train_size", train_size)

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    formula_vec = np.array([0.0 for k in range(herb_embeddings_dim)])
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    formula_len = len(herbs)
    for herb in herbs:
        if herb in herb_vector_map:
            herb_vector = herb_vector_map[herb]
            # print(formula_vec)
            # print(np.array(herb_vector))
            formula_vec = formula_vec + np.array(herb_vector)

    for j in range(herb_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(formula_vec[j] / formula_len)  # formula_vec[j]/ formula_len

# x = sp.csr_matrix((real_train_size, herb_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, herb_embeddings_dim))

y = []
for i in range(real_train_size):
    formula_meta = shuffle_formula_name_list[i]
    temp = formula_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)

# tx: feature vectors of test formulas, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    formula_vec = np.array([0.0 for k in range(herb_embeddings_dim)])
    formula_herbs = shuffle_formula_herbs_list[i + train_size]
    herbs = formula_herbs.split()
    formula_len = len(herbs)
    for herb in herbs:
        if herb in herb_vector_map:
            herb_vector = herb_vector_map[herb]
            formula_vec = formula_vec + np.array(herb_vector)

    for j in range(herb_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(formula_vec[j] / formula_len)  # formula_vec[j] / formula_len

# tx = sp.csr_matrix((test_size, herb_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, herb_embeddings_dim))

ty = []
for i in range(test_size):
    formula_meta = shuffle_formula_name_list[i + train_size]
    temp = formula_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> herbs

herb_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, herb_embeddings_dim))

for i in range(len(vocab)):
    herb = vocab[i]
    if herb in herb_vector_map:
        vector = herb_vector_map[herb]
        herb_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    formula_vec = np.array([0.0 for k in range(herb_embeddings_dim)])
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    formula_len = len(herbs)
    for herb in herbs:
        if herb in herb_vector_map:
            herb_vector = herb_vector_map[herb]
            formula_vec = formula_vec + np.array(herb_vector)

    for j in range(herb_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(formula_vec[j] / formula_len)  # formula_vec[j]/formula_len

for i in range(vocab_size):
    for j in range(herb_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(herb_vectors.item((i, j)))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, herb_embeddings_dim))

ally = []
for i in range(train_size):
    formula_meta = shuffle_formula_name_list[i]
    temp = formula_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
'''
formula-herb heterogeneous graph
'''

row = []
col = []
weight = []

# herb vector cosine similarity as weights
sum1 = 0
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in herb_vector_map and vocab[j] in herb_vector_map:
            vector_i = np.array(herb_vector_map[vocab[i]])
            vector_j = np.array(herb_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)

            if similarity > 0:
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
                sum1 += 1

            # if similarity > 0.9:
            #     # print(vocab[i], vocab[j], similarity, '\t')
            #     row.append(train_size + i)
            #     col.append(train_size + j)
            #     weight.append(similarity)

# formula herb frequency
# formula_herb_freq = {}
#
# for formula_id in range(len(shuffle_formula_herbs_list)):
#     formula_herbs = shuffle_formula_herbs_list[formula_id]
#     herbs = formula_herbs.split()
#     for herb in herbs:
#         herb_id = herb_id_map[herb]
#         formula_herb_str = str(formula_id) + ',' + str(herb_id)
#         if formula_herb_str in formula_herb_freq:
#             formula_herb_freq[formula_herb_str] += 1
#         else:
#             formula_herb_freq[formula_herb_str] = 1

sum2 = 0
pcdd_dict = {}   # Storage PHDD value
idf_dict = {}  # idf value of each herb
for i in range(len(shuffle_formula_herbs_list)):
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    formula_herb_set = set()
    PCDD = []
    for herb in herbs:
        if herb in formula_herb_set:
            continue
        j = herb_id_map[herb]
        # key = str(i) + ',' + str(j)
        # freq = formula_herb_freq[key]
        freq = pow(0.5, (len(herbs))/herb_avg_len[herb])

        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_formula_herbs_list) /
                  herb_formula_freq[vocab[j]])
        idf_dict[herb] = idf
        dependence = freq * idf
        PCDD.append(dependence)
        weight.append(dependence)
        sum2 += 1
        formula_herb_set.add(herb)
    pcdd_dict[shuffle_formula_herbs_list[i]]=PCDD


print("sum1+sum2", sum1+sum2)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
with open("../data/ind.{}.x".format(dataset), 'wb') as f:
    pkl.dump(x, f)

with open("../data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("../data/ind.{}.tx".format(dataset), 'wb') as f:
    pkl.dump(tx, f)

with open("../data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("../data/ind.{}.allx".format(dataset), 'wb') as f:
    pkl.dump(allx, f)

with open("../data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open("../data/ind.{}.adj".format(dataset), 'wb') as f:
    pkl.dump(adj, f)
