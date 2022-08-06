#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import Data, read_sessions, seq_augument, inputs_target_split
from model import *

def init_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
init_seed(2021)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='sample/diginetica/gowalla')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=0, help='l2 penalty')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--drop_prob', type=float, default=0.25, help='dropout')
parser.add_argument('--n_num', type=int, default=2, help='the neighbor number in the global graph')
parser.add_argument('--hop', type=int, default=1, help='the hop number in the global graph')
parser.add_argument('--neg_frac', type=int, default=128, help='the ratio of items selected as negative samples')
opt = parser.parse_args()
print(opt)

def check_neighbor_num(neighbors):
    a = np.sum(np.array(neighbors), -1)
    a = np.sign(a)
    print ('The percentage of items with neighbors  = ', np.sum(a) / a.shape[0])

def main():
    data_floder = '../datasets/'

    train_seqs = read_sessions(data_floder + opt.dataset + '/' + 'train.txt')
    test_seqs = read_sessions(data_floder + opt.dataset + '/' + 'test.txt')
    if opt.dataset == 'sample':
        train_seqs = train_seqs[:int(len(train_seqs)/10)]
        test_seqs = test_seqs[:int(len(test_seqs)/10)]
    train_seqs = seq_augument(train_seqs)
    test_seqs = seq_augument(test_seqs)
    print('num of training samples = ', len(train_seqs))
    print('num of test samples = ', len(test_seqs))
    
    train_data = inputs_target_split(train_seqs)
    test_data = inputs_target_split(test_seqs)

    train_lens = [len(seq) for seq in train_data[0]]
    test_lens = [len(seq) for seq in test_data[0]]
    print('max train len = ', max(train_lens))
    print('max test len = ', max(test_lens))

    with open(data_floder + opt.dataset + '/' + 'num_items.txt', 'r') as g:
        opt.n_node = int(g.readline()) + 1 # plus 1
    g_neighbors = pickle.load(open('graph/'+str(opt.dataset)+'/'+'g_neighbors_%d.txt'% opt.n_num,'rb'))
    check_neighbor_num(g_neighbors)

    l_neighbors_train = pickle.load(open('graph/'+str(opt.dataset)+'/'+'l_neighbors_train.txt','rb'))
    l_neighbors_test = pickle.load(open('graph/'+str(opt.dataset)+'/'+'l_neighbors_test.txt','rb'))

    train_data = Data(train_data, l_neighbors_train, g_neighbors, opt, shuffle=True)
    test_data = Data(test_data, l_neighbors_test, g_neighbors, opt, shuffle=False)

    model = trans_to_cuda(SessionGraph(opt, opt.n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data, opt)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
