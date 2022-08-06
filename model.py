#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import trans_to_cuda, trans_to_cpu

class GAT(Module):
    def __init__(self, opt=None):
        super(GAT, self).__init__()
        self.opt = opt
        self.query = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.key = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.value = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.W_att = nn.Linear(2*self.opt.hiddenSize, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(self.opt.drop_prob)

    def GATCell(self, nodes_hidden, neighbors_hidden, mask):
        neighbors_hidden = self.dropout(torch.cat([nodes_hidden.unsqueeze(2), neighbors_hidden], -2))
        mask = torch.cat([trans_to_cuda(torch.ones(mask.shape[:-1])).unsqueeze(-1), mask], -1)

        # transformation
        query = self.query(nodes_hidden)
        query = query.unsqueeze(2).repeat(1, 1, neighbors_hidden.shape[2], 1) # bs * seq_len * seq_len * latent_size
        key = self.key(neighbors_hidden)
        value = self.dropout(self.value(neighbors_hidden))
        # simlarity
        sim = self.W_att(torch.cat([query, key], -1)).squeeze(-1) # bs * seq_len * seq_len
        sim = self.leakyrelu(sim)
        sim = torch.exp(sim) * mask
        sim = sim / (torch.sum(sim, -1, keepdim=True) + 1e-16)
        # mulplication
        output = torch.matmul(sim.unsqueeze(2), value).squeeze(2)

        return output

    def forward(self, nodes_hidden, neighbors_hidden, mask):
        hidden = self.GATCell(nodes_hidden, neighbors_hidden, mask)
        return hidden

def cross_entropy_max(x, tar_idx, opt):
    idx0_temp = x[:, 0] # bs
    tar_score = x[torch.arange(x.shape[0]).long(), tar_idx] # bs

    x[torch.arange(x.shape[0]).long(), tar_idx] = idx0_temp
    # in prediction: minus 1 and remove target score minus 1
    neg_scores = x[:,1:].topk(int((opt.n_node-2)/opt.neg_frac))[0] # bs * neg_num
    scores = torch.cat([tar_score.unsqueeze(-1), neg_scores], -1)

    scores = torch.log_softmax(scores, -1) # bs * candidate_num
    loss_temp = -scores[:, 0] # bs

    loss = torch.mean(loss_temp) # 1
    return loss

class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(self.opt.n_node, self.opt.hiddenSize)

        # parameters for propagation
        self.localgnn_self = GAT(opt)
        self.localgnn_coatt = GAT(opt)
        self.globalgnn_self = GAT(opt)
        self.globalgnn_coatt = GAT(opt)
        self.local_concat = nn.Linear(2*self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.global_concat = nn.Linear(2*self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.dropout = nn.Dropout(self.opt.drop_prob)

        # parameters for combination
        self.combine_concat = nn.Linear(4*self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        
        # parameters for prediction
        self.linear_one = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.linear_two = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.linear_three = nn.Linear(self.opt.hiddenSize, 1, bias=True)
        self.predict_concat = nn.Linear(2*self.opt.hiddenSize, self.opt.hiddenSize, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.opt.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def propagation(self, nodes, l_neighbors, g_neighbors):
        local_mask = torch.sign(l_neighbors).float()
        global_mask = torch.sign(g_neighbors).float()

        nodes_hidden = self.embedding(nodes) # bs * seq_len * latent_size
        l_neighbors_self = self.embedding(l_neighbors) # bs * seq_len * l_len * latent_size
        g_neighbors_self = self.embedding(g_neighbors) # bs * seq_len * g_len * latent_size
        
        # generate affinity matrix for message aggregation
        # C_matrix: bs * seq_len * l_len * g_len
        C_matrix = torch.matmul(l_neighbors_self, g_neighbors_self.transpose(-2, -1))
        C_mask = local_mask.unsqueeze(-1) * global_mask.unsqueeze(2)
        C_mask = (1-C_mask) * -10000
        C_matrix = C_matrix + C_mask

        # incorporate gobal information for local graph
        alpha_local = torch.softmax(C_matrix, -1)
        l_neighbors_coatt = torch.matmul(alpha_local, g_neighbors_self)
        
        # incorporate local information for global graph
        alpha_global = torch.softmax(C_matrix.transpose(-2, -1), -1)
        g_neighbors_coatt = torch.matmul(alpha_global, l_neighbors_self)

        local_hidden_self = self.localgnn_self(nodes_hidden, l_neighbors_self, local_mask)
        global_hidden_self = self.localgnn_self(nodes_hidden, g_neighbors_self, global_mask)
        local_hidden_coatt = self.localgnn_coatt(nodes_hidden, l_neighbors_coatt, local_mask)
        global_hidden_coatt = self.globalgnn_coatt(nodes_hidden, g_neighbors_coatt, global_mask)

        local_hidden = self.local_concat(torch.cat([local_hidden_self, local_hidden_coatt], -1))
        global_hidden = self.global_concat(torch.cat([global_hidden_self, global_hidden_coatt], -1))

        return local_hidden, global_hidden

    def combination(self, local_hidden, global_hidden, mask):
        C_matirx = torch.matmul(local_hidden, global_hidden.transpose(-2, -1)) # bs * seq_len * seq_len
        C_mask = mask.unsqueeze(2) * mask.unsqueeze(1) # bs * seq_len * seq_len
        C_mask = (1-C_mask) * -10000
        C_matirx = C_matirx + C_mask

        # local information from C
        local_alpha = torch.softmax(C_matirx, -1) # bs * seq_len * seq_len
        local_from_C = torch.matmul(local_alpha, global_hidden) # bs * seq_len * latent_size
        # global information from C
        global_alpha = torch.softmax(C_matirx.transpose(-2, -1), -1) # bs * seq_len * seq_len
        global_from_C = torch.matmul(global_alpha, local_hidden) # bs * seq_len * latent_size

        local_hidden, global_hidden = self.dropout(local_hidden), self.dropout(global_hidden)
        local_from_C, global_from_C = self.dropout(local_from_C), self.dropout(global_from_C)

        combination = torch.cat([local_hidden, local_from_C, global_hidden, global_from_C], -1)

        output = self.combine_concat(combination)
        return output

    def prediction(self, hidden, mask):
        s = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(s).view(s.shape[0], 1, s.shape[1])
        q2 = self.linear_two(hidden)  # bs * seq_len * latent_size
        alpha = self.linear_three(torch.sigmoid(q1+q2)).squeeze(-1)
        alpha = torch.exp(alpha) * mask.view(mask.shape[0], -1).float()# bs * seq_len
        alpha = alpha / torch.sum(alpha, -1, keepdim=True)
        l = torch.sum(alpha.unsqueeze(-1) * hidden, 1)

        a = self.predict_concat(torch.cat([l, s], -1))

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        
        return scores

def forward(model, i, data):
    inputs, l_neighbors, g_neighbors, mask, targets = data.get_slice(i)

    inputs = trans_to_cuda(torch.Tensor(inputs).long())
    l_neighbors = trans_to_cuda(torch.Tensor(l_neighbors).long())
    g_neighbors = trans_to_cuda(torch.Tensor(g_neighbors).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    # information propagation
    local_hidden, global_hidden = model.propagation(inputs, l_neighbors, g_neighbors)
    hidden = model.combination(local_hidden, global_hidden, mask)
    # prediction
    scores = model.prediction(hidden, mask)

    return targets, scores

def train_test(model, train_data, test_data, opt):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.opt.batchSize)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = cross_entropy_max(scores, targets - 1, opt)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.opt.batchSize)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
