import pickle
import argparse
import datetime
import numpy as np

import scipy.sparse as sp
from scipy.sparse import csr_matrix, dok_matrix
import os
import heapq
import copy
from utils import read_sessions

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='sample/diginetica/gowalla')
parser.add_argument('--n_num', type=int, default=2, help='number of neighbors in global graph')
opt = parser.parse_args()

data_floder = '../datasets/'
seq = read_sessions(data_floder + opt.dataset + '/' + 'train.txt')
with open(data_floder + opt.dataset + '/' + 'num_items.txt', 'r') as f:
    opt.n_node = int(f.readline())
print ('begin constraction time = ', datetime.datetime.now())

adj_matrix_dok = dok_matrix((opt.n_node, opt.n_node), dtype=np.int32)
for user_items in seq:
    for i in range(len(user_items)):
        if i == len(user_items) - 1:
            break
        elif user_items[i] != user_items[i+1]:
            adj_matrix_dok[ user_items[i], user_items[i+1] ] += 1
            adj_matrix_dok[ user_items[i+1], user_items[i] ] += 1
        else:
            continue

adj_matrix = dok_matrix.tocsr(adj_matrix_dok)
adj_matrix = adj_matrix.toarray()
adj_matrix = csr_matrix(adj_matrix)

neighbors_idx = []
neighbors_value = []
neighbors_idx.append([0] * opt.n_num)
neighbors_value.append([0] * opt.n_num)

for item in range(opt.n_node):
    begin = adj_matrix.indptr[item]
    end = adj_matrix.indptr[item+1]
    adj_index = adj_matrix.indices[begin:end]
    adj_value = adj_matrix.data[begin:end]
    adj_value = list(adj_value)

    if adj_value != []:
        # select the top opt.n_num as the neighbors
        data = heapq.nlargest(opt.n_num, enumerate(adj_value), key=lambda x:x[1])
        max_idx, vals = zip(*data)

        max_index = [adj_index[i] for i in max_idx]
        max_value = list(vals)        
        # padding using 0 for both index and value
        max_index = max_index + [0] * (opt.n_num - len(max_index))
        max_value = max_value + [0] * (opt.n_num - len(max_value))
    else:
        # padding using 0 for both index and value
        max_index = [0] * opt.n_num
        max_value = [0] * opt.n_num
    
    neighbors_idx.append(max_index)
    neighbors_value.append(max_value)

neighbors_info = [neighbors_idx, neighbors_value]
print ('end constraction time = ', datetime.datetime.now())

if not os.path.exists('graph/'):
    os.mkdir('graph/')
if not os.path.exists('graph/'+opt.dataset):
    os.mkdir('graph/'+opt.dataset)
pickle.dump(neighbors_idx, open('graph/'+opt.dataset+'/'+'g_neighbors_%d.txt' % opt.n_num, 'wb'))

print ('done')
