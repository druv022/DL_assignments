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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.w_hx = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        nn.init.normal_(self.w_hx, 0.0, 1e-2)
        self.w_hh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        nn.init.normal_(self.w_hh, 0.0, 1e-2)
        #self.bias_h = nn.Parameter(torch.zeros(num_hidden))
        self.w_ph = nn.Parameter(torch.FloatTensor(num_classes, num_hidden))
        nn.init.normal_(self.w_ph, 0.0, 1e-2)
        #self.bias_p = nn.Parameter(torch.zeros(num_classes))

        # initialization for first time step
        self.h_init = torch.zeros(batch_size,num_hidden).to(device)

        # non-linearity
        self.tanh = nn.Tanh()


    def forward(self, x):
        # Implementation here ...
        # batch size, length of sequence, embedding dimension/one-hot vector dim
        b, l = x.size()

        assert l == self.seq_length ,"Sequence length mismatch"
        assert b == self.batch_size, "Batch size mismatch"

        h_t = []
        p_t = []
        
        # layer index starts from 1
        for layer in torch.arange(1, self.seq_length+1):
            # initialization at first time step
            if layer == 1:
                h_t.append(self.h_init)

            h_t.append(self.tanh(torch.mm(torch.unsqueeze(x[:,int(layer)-1],1),self.w_hx.transpose(1,0)) + 
                    torch.mm(h_t[int(layer)-1], self.w_hh)))# + self.bias_h)
            p_t.append(torch.mm(h_t[int(layer)],self.w_ph.transpose(1,0)))# + self.bias_p)
        
        
        return p_t[-1]



                    