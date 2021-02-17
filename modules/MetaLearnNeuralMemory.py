import math, copy, sys, logging, json, time, random, os, string, pickle, re

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MNMp(nn.Module):

    def __init__(self, dim_hidden, n_heads = 4):
        
        """ dim_hidden is the hidden size of the LSTM controller,
            the Memory Network, and the interaction vectors
            n_heads is the number of interaction heads """

        super(MNMp, self).__init__()
        
        self.dim_hidden = dim_hidden
        self.n_heads = n_heads
        
        self.control = nn.LSTMCell(dim_hidden*2, dim_hidden)
        
        dim_concat_interact = dim_hidden*n_heads*3 + dim_hidden
        self.interaction = nn.Linear(dim_hidden, dim_concat_interact)
        self.memfunc = FFMemoryLearned(dim_hidden)
        self.kv_rate = nn.Linear(dim_hidden, 1)
        self.read_out = nn.Linear(dim_hidden+dim_hidden, dim_hidden)
        
        self.v_r = None
        self.h_lstm = None
        self.c_lstm = None
            
    def initialize_v_h_c(self, batch_size):
        
            self.v_r = torch.zeros((batch_size, self.dim_hidden)).float()
            self.h_lstm = torch.zeros((batch_size, self.dim_hidden)).float()
            self.c_lstm = torch.zeros((batch_size, self.dim_hidden)).float()
            
            if next(self.parameters()).is_cuda:
                self.v_r = self.v_r.cuda()
                self.h_lstm = self.h_lstm.cuda()
                self.c_lstm = self.c_lstm.cuda()
            
    def forward(self, x):
        """ the input must have shape (batch_size, emb_dim) because it will be 
        concatenated with self.v_r of the same shape """

        self.initialize_v_h_c(x.shape[0])
        x = x.squeeze(1)
        self.h_lstm, self.c_lstm = self.control(torch.cat([x, self.v_r], dim=1), 
                                                (self.h_lstm, self.c_lstm))
        
        int_vecs = torch.tanh(self.interaction(self.h_lstm))
        beta_, n_k_v = torch.split(int_vecs, 
                                   [self.dim_hidden, 
                                   self.dim_hidden*self.n_heads*3],
                                   dim=1)  
        
        beta = torch.sigmoid(self.kv_rate(beta_)) #(batch_size,1)
        n_k_v = n_k_v.view(n_k_v.shape[0], self.n_heads, -1).contiguous()
        k_w, v_w, k_r = torch.chunk(n_k_v, 3, dim=2)
        reconst_loss, reconst_loss_init = self.memfunc.update(k_w, v_w, 
                                                                beta_rate=beta)
        self.v_r = self.memfunc.read(k_r)
        h_lstm = self.read_out(torch.cat([self.h_lstm, self.v_r], dim=1))

        return h_lstm.unsqueeze(1), reconst_loss, reconst_loss_init 
        #return x, torch.Tensor([0]), torch.Tensor([0]) #

class FFMemoryLearned(nn.Module):
    
    def __init__(self, dim_hidden):
        
        super(FFMemoryLearned, self).__init__()
        
        self.l1 = nn.Linear(dim_hidden, dim_hidden).weight.data
        self.l2 = nn.Linear(dim_hidden, dim_hidden).weight.data
        self.l3 = nn.Linear(dim_hidden, dim_hidden).weight.data
        
        self.Ws = [self.l1, self.l2, self.l3]
        self.Ws_temp = None
        
        self.expected_activation = nn.Linear(dim_hidden, 3*dim_hidden)

    def if_cuda(self):
        if next(self.parameters()).is_cuda:
            self.l1 = self.l1.cuda()
            self.l2 = self.l2.cuda()
            self.l3 = self.l3.cuda()
        self.Ws = [self.l1, self.l2, self.l3]

    def detach_mem(self):
        for W_l in self.Ws:
          W_l.detach()
        self.Ws_temp = None

    def forward(self, key):
        self.if_cuda()
        
        if self.Ws_temp is None: # is reset at beginning of every new sequence 
          self.Ws_temp = []
          for W_l in self.Ws:
            W = W_l.unsqueeze(dim=0).expand((key.shape[0], W_l.shape[0], W_l.shape[1]))
            self.Ws_temp.append(W)
            
        activations = [] # hidden activations 
        a = key
        
        for W_l in self.Ws_temp[:-1]:
          a = torch.matmul(a, W_l.transpose(1,2))
          a = torch.sigmoid(a)
          activations.append(a)
        
        value = torch.matmul(a, self.Ws_temp[-1].transpose(1,2))

        return value, activations

    def mse_loss(self, y_pred, y):
        diff = y_pred - y
        diff = diff.view(-1)
        mse = diff.dot(diff)/diff.size()[0]
        return mse

    def read(self, k, weights=None, avg=True):

        v, h_acts = self.forward(k)

        if weights is not None:
          v *= weights
        if avg:
          v = v.mean(dim=1)
        return v

    def update(self, k_w, v_w, beta_rate=0.1):  

        v_w_approx, activations = self.forward(k_w)
        
        v_w_approx = v_w_approx.contiguous()
        v_w = v_w.contiguous()
        
        reconst_loss_init = self.mse_loss(v_w_approx.view(-1, v_w_approx.shape[2]), 
                                           v_w.view(-1, v_w.shape[2]))
        
        z_pr = self.expected_activation(v_w.view(-1, v_w.shape[2]))
        z_pr = z_pr.view(v_w.shape[0], v_w.shape[1], -1)
        z_pr = torch.chunk(z_pr, 3, dim=2)
        
        if len(beta_rate.shape) < 3:
            beta_rate = beta_rate.unsqueeze(1)

        z2 = activations + [v_w_approx]
        z1 = [k_w] + activations
        Ws_t = []
        
        for W_l, z2_, z1_, z_pr in reversed(list(zip(self.Ws_temp, 
                                             z2, z1, z_pr))):
            
            z1_ = z1_*beta_rate.expand(z1_.shape)
            diff = z2_ - z_pr #(batchsize, heads, dim_hidden)
            diff = diff*(2./ (diff.shape[1]*diff.shape[2]))
            W_l = W_l - torch.matmul(diff.transpose(1,2), z1_)#- 0.0001*W_l 0.1
            Ws_t.insert(0, W_l)

        self.Ws_temp[:] = Ws_t
        
        # Run memory forward again after memory update 
        v_w_approx, activations = self.forward(k_w)
        reconst_loss = self.mse_loss(v_w_approx.view(-1, v_w_approx.shape[2]), 
                                      v_w.view(-1, v_w.shape[2]))
        
        return reconst_loss, reconst_loss_init