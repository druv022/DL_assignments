# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# removing part3 
from dataset import TextDataset
from model import TextGenerationModel
import os
################################################################################
    # def sample(h, seed_ix, n):
    #   """ 
    #   sample a sequence of integers from the model 
    #   h is memory state, seed_ix is seed letter for first time step
    #   """
    #   x = np.zeros((vocab_size, 1))
    #   x[seed_ix] = 1
    #   ixes = []
    #   for t in xrange(n):
    #     h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    #     y = np.dot(Why, h) + by
    #     p = np.exp(y) / np.sum(np.exp(y))
    #     ix = np.random.choice(range(vocab_size), p=p.ravel())
    #     x = np.zeros((vocab_size, 1))
    #     x[ix] = 1
    #     ixes.append(ix)
    # return ixes


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize the dataset and data loader (note the +1)
    txt_file = '/media/druv022/Data1/git/DL_assignments/assignment_2/part3/assets/book_EN_grimms_fairy_tails.txt' #REMOVE ME
    dataset = TextDataset(txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, lstm_num_hidden=config.lstm_num_hidden, 
                lstm_num_layers=config.lstm_num_layers, device= device)  # fixme
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduce=True)  # fixme
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)  # fixme
    step = 1
    for epochs in range(30):

        for (batch_inputs, batch_targets) in data_loader:

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            if config.batch_size!=batch_inputs.size()[0]:
                print("batch mismatch")
                continue 
            batch_inputs_onehot = torch.zeros(batch_inputs.size()[0], config.seq_length, dataset.vocab_size).scatter_(2,torch.unsqueeze(batch_inputs,2),1)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()

            model.train()
            y = model(batch_inputs_onehot.to(device))

            loss = criterion(y.transpose(2,1),batch_targets)
            loss.backward()
            optimizer.step()
            _, predictions = torch.max(y, dim=2, keepdim=True)
            predictions = (predictions.squeeze(-1) == batch_targets).float()

            accuracy = torch.mean(predictions)  # fixme

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {}/{}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))
                

            if step == config.sample_every:
                # Generate some sentences by sampling from the model
                # if not os.path.isdir(config.summary_path):
                #    os.makedirs(config.summary_path) 
               
                # with open(os.path.join(config.summary_path,"index.txt"), "a+") as myfile:
                #     myfile.write("--------------",str(step),"----------------")

                pass
                

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
            step+=1
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on") FIX ME
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
