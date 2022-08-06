#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    sessions = (sessions).tolist()
    return sessions

def seq_augument(seqs):
    aug_seqs = []
    for seq in seqs:
        for i in range(2, len(seq)+1):
            aug_seqs.append((np.array(seq[:i])+1).tolist()) # index from 1
    return aug_seqs

def inputs_target_split(seqs):
    inputs, targets = [], []
    for seq in seqs:
        inputs.append(seq[:-1])
        targets.append(seq[-1])
    return [inputs, targets]

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

class Data():
    def __init__(self, data, l_neighbors, g_neighbors, opt, shuffle=False):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.l_neighbors = np.asarray(l_neighbors)
        self.g_neighbors = np.asarray(g_neighbors)
        self.opt = opt

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.l_neighbors = self.l_neighbors[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        # local neighbors
        l_neighbors = self.l_neighbors[i]
        # global neighbors
        g_neighbors = self.g_neighbors[np.array(inputs)]
        last_neighbors = g_neighbors
        for i in range(self.opt.hop-1):
            last_neighbors = self.g_neighbors[last_neighbors]
            last_neighbors = np.reshape(last_neighbors, (g_neighbors.shape[0], g_neighbors.shape[1], -1))
            g_neighbors = np.concatenate([g_neighbors, last_neighbors], -1)

        return inputs, l_neighbors, g_neighbors, mask, targets
