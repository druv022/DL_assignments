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

        self.unfold_layers = {}

        # adding input to hidden layer
        self.unfold_layers['input_hidden'] = nn.Linear(input_dim, num_hidden)
        self.unfold_layers['input_hidden'].weight.data.normal_(0.0,1e-2)
        self.unfold_layers['input_hidden'].bias.data.fill_(0.0)

        # adding recurrent layer
        self.unfold_layers['hidden'] = nn.Linear(num_hidden, num_hidden)
        self.unfold_layers['hidden'].weight.data.normal_(0.0,1e-2)
        self.unfold_layers['hidden'].bias.data.fill_(0.0)

        # adding hidden to output layer
        self.unfold_layers['hidden_output']  = nn.Linear(num_hidden, num_classes)
        self.unfold_layers['hidden_output'].weight.data.normal_(0.0,1e-2)
        self.unfold_layers['hidden_output'].bias.data.fill_(0.0)

        # initialization for first time step
        self.h_init = torch.zeros(batch_size,num_hidden).to(device)
        self.b_init = torch.zeros(batch_size,num_hidden).to(device)

        # non-linearity
        self.tanh = nn.Tanh()

        self.modules = nn.ModuleDict(self.unfold_layers)


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

            h_t.append(self.tanh(self.unfold_layers['input_hidden'](torch.unsqueeze(x[:,int(layer)-1],1)) + 
                    self.unfold_layers['hidden'](h_t[int(layer)-1])))
            p_t.append(self.unfold_layers['hidden_output'](h_t[int(layer)]))
        
        
        return p_t[-1]



                    