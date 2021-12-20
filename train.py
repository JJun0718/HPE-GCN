# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from models.gcn import GCN


from config import CONFIG
cfg = CONFIG()
dataset = cfg.dataset

# Set random seed
seed = random.randint(1, 200)
# seed = 118
print(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

# Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    cfg.dataset)

features = sp.identity(features.shape[0])  # featureless
# Some preprocessing
features = preprocess_features(features)
if cfg.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif cfg.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, cfg.max_degree)
    num_supports = 1 + cfg.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(cfg.model))

# Define placeholders
t_features = torch.from_numpy(features)
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

t_support = []
for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]))

# if torch.cuda.is_available():
#     model_func = model_func.cuda()
#     t_features = t_features.cuda()
#     t_y_train = t_y_train.cuda()
#     t_y_val = t_y_val.cuda()
#     t_y_test = t_y_test.cuda()
#     t_train_mask = t_train_mask.cuda()
#     tm_train_mask = tm_train_mask.cuda()
#     for i in range(len(support)):
#         t_support = [t.cuda() for t in t_support if True]

model = model_func(input_dim=features.shape[0], support=t_support, num_classes=y_train.shape[1])
# model = model_func(input_dim=features.shape[0], num_classes=y_train.shape[1])

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate)
# Define model evaluation function
def evaluate(features, labels, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
        
    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)

val_losses = []

Loss_list = []
Accuracy_list = []

# Train model
for epoch in range(cfg.epochs):

    t = time.time()
    
    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])    
    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()

    Loss_list.append(loss)
    Accuracy_list.append(acc)
        
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_losses.append(val_loss)

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping+1):-1]):
        final_epoch = epoch
        print_log("Early stopping...")
        break

print_log("Optimization Finished!")

def plot_acc_loss(final_epoch):
    x1 = range(0, final_epoch+1)
    x2 = range(0, final_epoch+1)
    y1 = Accuracy_list
    y2 = Loss_list

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("Result/accuracy_loss.jpg")


# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(np.argmax(labels[i]))

# show labels
target_names = []
with open('./data/corpus/' + dataset + '_labels.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        target_names.append(line.strip())

print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, target_names=target_names, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# formula and herb embeddings
tmp = model.layer1.embedding.numpy()
herb_embeddings = tmp[train_size: adj.shape[0] - test_size]
train_formula_embeddings = tmp[:train_size]  # include val formulas
test_formula_embeddings = tmp[adj.shape[0] - test_size:]

print_log('Embeddings:')
print_log('\rherb_embeddings:'+str(len(herb_embeddings)))
print_log('\rTrain_formula_embeddings:'+str(len(train_formula_embeddings))) 
print_log('\rTest_formula_embeddings:'+str(len(test_formula_embeddings))) 
print_log('\rherb_embeddings:') 
print(herb_embeddings)

with open('./data/corpus/' + dataset + '_vocab.txt', 'r', encoding='utf8') as f:
    herbs = f.readlines()

vocab_size = len(herbs)
herb_vectors = []
for i in range(vocab_size):
    herb = herbs[i].strip()
    herb_vector = herb_embeddings[i]
    herb_vector_str = ' '.join([str(x) for x in herb_vector])
    herb_vectors.append(herb + ' ' + herb_vector_str)

herb_embeddings_str = '\n'.join(herb_vectors)
with open('./data/' + dataset + '_herb_vectors.txt', 'w', encoding='utf8') as f:
    f.write(herb_embeddings_str)



formula_vectors = []
formula_id = 0
for i in range(train_size):
    formula_vector = train_formula_embeddings[i]
    formula_vector_str = ' '.join([str(x) for x in formula_vector])
    formula_vectors.append('formula_' + str(formula_id) + ' ' + formula_vector_str)
    formula_id += 1

for i in range(test_size):
    formula_vector = test_formula_embeddings[i]
    formula_vector_str = ' '.join([str(x) for x in formula_vector])
    formula_vectors.append('formula_' + str(formula_id) + ' ' + formula_vector_str)
    formula_id += 1

formula_embeddings_str = '\n'.join(formula_vectors)
with open('./data/' + dataset + '_formula_vectors.txt', 'w', encoding='utf8') as f:
    f.write(formula_embeddings_str)



# plot_acc_loss(final_epoch)
