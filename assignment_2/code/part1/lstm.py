################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.w_gx = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        nn.init.normal_(self.w_gx, 0.0, 1e-2)
        self.w_gh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        nn.init.normal_(self.w_gh, 0.0, 1e-2)
        self.w_ix = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        nn.init.normal_(self.w_ix, 0.0, 1e-2)
        self.w_ih = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        nn.init.normal_(self.w_ih, 0.0, 1e-2)
        self.w_fx = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        nn.init.normal_(self.w_fx, 0.0, 1e-2)
        self.w_fh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        nn.init.normal_(self.w_fh, 0.0, 1e-2)
        self.w_ox = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        nn.init.normal_(self.w_ox, 0.0, 1e-2)
        self.w_oh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        nn.init.normal_(self.w_oh, 0.0, 1e-2)
        self.w_ph = nn.Parameter(torch.FloatTensor(num_classes, num_hidden))
        nn.init.normal_(self.w_ph, 0.0, 1e-2)
        self.bias_g = nn.Parameter(torch.zeros(num_hidden))
        self.bias_i = nn.Parameter(torch.zeros(num_hidden))
        self.bias_f = nn.Parameter(torch.zeros(num_hidden))
        self.bias_o = nn.Parameter(torch.zeros(num_hidden))
        self.bias_p = nn.Parameter(torch.zeros(num_classes))

        # initialization for first time step
        self.h_init = torch.zeros(batch_size,num_hidden).to(device)
        self.c_init = torch.zeros(batch_size,num_hidden).to(device)

        # non-linearity
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Implementation here ...
        b, l = x.size()

        h_t = []
        c_t = []

        assert l == self.seq_length ,"Sequence length mismatch"
        assert b == self.batch_size, "Batch size mismatch"

        for layer in torch.arange(1, self.seq_length+1):
            if int(layer) == 1:
                h_t.append(self.h_init)
                c_t.append(self.c_init)

            x_data = torch.unsqueeze(x[:,int(layer)-1],1)
            g_t = self.tanh(torch.mm(x_data, self.w_gx.transpose(1,0)) + torch.mm(h_t[int(layer)-1],self.w_gh.transpose(1,0)) + self.bias_g)
            i_t = self.sigmoid(torch.mm(x_data, self.w_ix.transpose(1,0)) + torch.mm(h_t[int(layer)-1],self.w_ih.transpose(1,0)) + self.bias_i)
            f_t = self.sigmoid(torch.mm(x_data, self.w_fx.transpose(1,0)) + torch.mm(h_t[int(layer)-1],self.w_fh.transpose(1,0)) + self.bias_f)
            o_t = self.sigmoid(torch.mm(x_data, self.w_ox.transpose(1,0)) + torch.mm(h_t[int(layer)-1],self.w_oh.transpose(1,0)) + self.bias_o)
            c_t.append(g_t*i_t + c_t[int(layer)-1]*f_t)
            h_t.append(self.tanh(c_t[int(layer)])*o_t)

            p_t = torch.mm(h_t[int(layer)],self.w_ph.transpose(1,0)) + self.bias_p
        
        return p_t
