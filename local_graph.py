import numpy as np
import argparse
import pickle
import os
from utils import read_sessions, seq_augument, inputs_target_split, data_masks

def filter_lens(lens):
    max_len = max(lens)
    len_num = np.zeros(max_len+1,)
    for len_ in lens:
        len_num[len_] += 1
    
    print ('len_num = ', len_num)
    # remove the padding 0
    total_num = len(lens) - len_num[0]
    print ('total_num = ', total_num)
    cur_num = 0
    for len_ in range(1, max_len+1):
        cur_num += len_num[len_]
        if cur_num > total_num * 0.95:
            return len_

def find_l_neighbors(data):
    inputs = data[0]
    inputs, mask, len_max = data_masks(inputs, [0])
    inputs = np.asarray(inputs)

    n_node = []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)

    l_neighbors, lens = [], []
    for u_input in inputs:
        node = np.unique(u_input)
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input)-1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
            u_A[v][u] = 1
        for i in np.arange(len(node)):
            u_A[i][i] = 0

        neighbors = []
        for item in u_input:
            node_idx = np.where(node == item)
            index = np.where(u_A[node_idx][0] == 1)
            neighbors.append(list(node[index]))
            lens.append(len(list(node[index])))
        l_neighbors.append(neighbors)

    #max_len = max(lens)
    max_len = filter_lens(lens)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            l_neighbors[i][j] = l_neighbors[i][j][:max_len] + [0]*(max_len-len(l_neighbors[i][j]))

    return l_neighbors

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='sample/diginetica/gowalla')
opt = parser.parse_args()

data_floder = '../datasets/'
train_seqs = read_sessions(data_floder + opt.dataset + '/' + 'train.txt')
test_seqs = read_sessions(data_floder + opt.dataset + '/' + 'test.txt')
if opt.dataset == 'sample':
    train_seqs = train_seqs[:int(len(train_seqs)/10)]
    test_seqs = test_seqs[:int(len(test_seqs)/10)]
train_seqs = seq_augument(train_seqs)
test_seqs = seq_augument(test_seqs)

train_data = inputs_target_split(train_seqs)
test_data = inputs_target_split(test_seqs)

train_l_neighbors = find_l_neighbors(train_data)
test_l_neighbors = find_l_neighbors(test_data)

train_l_neighbors = np.array(train_l_neighbors)
test_l_neighbors = np.array(test_l_neighbors)

print (np.array(train_l_neighbors).shape)
print (np.array(test_l_neighbors).shape)

if not os.path.exists('graph/'):
    os.mkdir('graph/')
if not os.path.exists('graph/'+opt.dataset):
    os.mkdir('graph/'+opt.dataset)
pickle.dump(train_l_neighbors, open('graph/'+opt.dataset+'/'+'l_neighbors_train.txt', 'wb'))
pickle.dump(test_l_neighbors, open('graph/'+opt.dataset+'/'+'l_neighbors_test.txt', 'wb'))

print ('done')
