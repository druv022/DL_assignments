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
import shutil
import matplotlib.pyplot as plt 


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize the dataset and data loader (note the +1)
    # txt_file = '/media/druv022/Data1/git/DL_assignments/assignment_2/part3/assets/book_EN_democracy_in_the_US.txt' #REMOVE ME
    txt_file = '/media/druv022/Data1/git/DL_assignments/assignment_2/part3/assets/book_EN_grimms_fairy_tails.txt' #REMOVE ME
    dataset = TextDataset(txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, lstm_num_hidden=config.lstm_num_hidden, 
                lstm_num_layers=config.lstm_num_layers, device= device)  # fixme
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.NLLLoss()  # fixme
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    logSoftmax = torch.nn.LogSoftmax(dim=2)
    softmax = torch.nn.Softmax(dim=1)
    step = 1

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    # Resume checkopint 
    if config.resume:
        if os.path.isfile(config.resume):
            print("Loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("Checkpoint loaded '{}', steps {}".format(config.resume, checkpoint['step']))

    # configure sampling folder and file
    if not os.path.isdir(config.summary_path):
        os.makedirs(config.summary_path)

    if config.sampling == "greedy":
        f_w = open(os.path.join(config.summary_path,"sampled_"+config.sampling+"_"+config.eval_sampling+".txt"),"w+")
        f_name = '_'+config.sampling
    else:
        f_w = open(os.path.join(config.summary_path,"sampled_"+config.sampling+"_"+config.eval_sampling+"_"+str(config.temp)+".txt"),"w+")
        f_name = '_'+config.sampling+"_"+str(config.temp)

    best_accuracy = 0.0


    for epochs in range(30):

        for (batch_inputs, batch_targets) in data_loader:

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            if config.batch_size!=batch_inputs.size()[0]:
                print("batch size mismatch! Skipping")
                continue
            # One-hot vector
            batch_inputs_onehot = torch.zeros(batch_inputs.size()[0], config.seq_length, dataset.vocab_size).scatter_(2,torch.unsqueeze(batch_inputs,2),1)
            batch_inputs_onehot, batch_targets = batch_inputs_onehot.to(device),batch_targets.to(device)
            optimizer.zero_grad()

            model.train()
            model.init_hidden(config.batch_size)
            y = model(batch_inputs_onehot)

            if config.sampling == "greedy":
                y = logSoftmax(y)
            else:
                y = logSoftmax(y/config.temp)

            loss = criterion(y.transpose(2,1),batch_targets)
            # backpropagate
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # evaluate and save model
            _, predictions = torch.max(y, dim=2, keepdim=True)
            predictions = (predictions.squeeze(-1) == batch_targets).float()

            accuracy = torch.mean(predictions)  # fixme

            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy
            
                save_checkpoint({
                    'epoch': epochs + 1,
                    'step': step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler':lr_scheduler.state_dict(),
                    'accuracy': accuracy
                }, is_best, filename_add=f_name)

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

            # if step == config.sample_every:
            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model

                rand_char_index = torch.randint(0,dataset.vocab_size,(1,)).long()
                rand_char_onehot = torch.zeros(1, 1, dataset.vocab_size).scatter_(2,torch.unsqueeze(torch.unsqueeze(rand_char_index,1),2),1)

                chars_ix = [rand_char_index.item()]
                model.eval()
                model.init_hidden(1)
                for _ in torch.arange(1,config.seq_length):
                    out = model(rand_char_onehot.to(device))
                    
                    if config.eval_sampling == 'greedy':
                        _, out = torch.max(out, dim=2)
                    elif config.eval_sampling == 'random':
                        out = out.squeeze(0)
                        out = torch.multinomial(softmax(out/config.temp),1)
                    else:
                        print("This Sampling is not implemented.")

                    chars_ix.append(out.item())
                    rand_char_onehot = torch.zeros(1, 1, dataset.vocab_size).to(device).scatter_(2,torch.unsqueeze(out,2),1)

                sentence = dataset.convert_to_string(chars_ix)

                f_w.write(str(step)+": "+sentence+"\n")
                print(sentence)               

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

            step+=1

        save_checkpoint({
            'epoch': epochs + 1,
            'step': step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler':lr_scheduler.state_dict(),
            'accuracy': accuracy
        }, is_best, filename_add=f_name)

        if step > config.train_steps:
            break
    print('Done training.')
    f_w.close()


# save checkpoint
def save_checkpoint(state, is_best, filename_add=''):
    filename = 'checkpoint'+filename_add+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best'+filename_add+'.pth.tar')

 ################################################################################
 ################################################################################
 #### BONUS #####

def evaluate(config):
    
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    # np.random.seed(45)
    # torch.manual_seed(45)

    # Initialize the dataset
    # txt_file = '/media/druv022/Data1/git/DL_assignments/assignment_2/part3/assets/book_EN_democracy_in_the_US.txt' #REMOVE ME
    txt_file = '/media/druv022/Data1/git/DL_assignments/assignment_2/part3/assets/book_EN_grimms_fairy_tails.txt' #REMOVE ME
    dataset = TextDataset(txt_file, config.seq_length)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, lstm_num_hidden=config.lstm_num_hidden, 
                lstm_num_layers=config.lstm_num_layers, device= device)  # fixme
    model.to(device)

    softmax = torch.nn.Softmax(dim=1)

    # Load Parameters
    if config.best_m:
        if os.path.isfile(config.best_m):
            print("Loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("Checkpoint loaded '{}', steps {}".format(config.resume, checkpoint['step']))

    # configure sampling folder and file
    if not os.path.isdir(config.summary_path):
        os.makedirs(config.summary_path)

    if config.eval_sampling == "greedy":
        f_w = open(os.path.join(config.summary_path,"Bonus_sampled_"+config.sampling+".txt"),"w+")
    else:
        f_w = open(os.path.join(config.summary_path,"Bonus_sampled_"+config.sampling+"_"+str(config.temp)+".txt"),"w+")

    if len(config.text) == 0: 
        rand_char_index = torch.randint(0,dataset.vocab_size,(1,)).long()
        char_onehot = torch.zeros(1, 1, dataset.vocab_size).scatter_(2,torch.unsqueeze(torch.unsqueeze(rand_char_index,1),2),1)
        chars_ix = [rand_char_index.item()]
    else:
        chars_ix = dataset.convert_to_index(config.text)
        char_onehot = torch.zeros(1, config.seq_length, dataset.vocab_size).scatter_(2,torch.unsqueeze(torch.unsqueeze(torch.tensor(chars_ix),0),2),1)
    
    model.eval()
    model.init_hidden(1)
    for indx in torch.arange(1,config.extnd_length):
        out = model(char_onehot.to(device))
        
        if config.eval_sampling == 'greedy':
            _, out = torch.max(out, dim=2)
        elif config.eval_sampling == 'random':
            out = out.squeeze(0)
            out = torch.multinomial(softmax(out/config.temp),1)
        else:
            print("This Sampling is not implemented.")

        seq_length = out.shape[0] - 1
        
        chars_ix.append(out.squeeze(0)[-1].item())
        item_t = torch.tensor([out[seq_length,-1]]).to(config.device)
        char_onehot = torch.zeros(1, 1, dataset.vocab_size).to(device).scatter_(2,torch.unsqueeze(torch.unsqueeze(item_t,1),2),1)

        # check intermediate transition
        if indx % 30 == 0:
            print(dataset.convert_to_string(chars_ix))

    sentence = dataset.convert_to_string(chars_ix)

    f_w.write(sentence)
    print(sentence)

    f_w.close()


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
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction') #0.96
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    # New mics params
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--resume', type=str, default='checkpoint_greedy.pth.tar', help="Path to latest checkpoint")
    parser.add_argument('--sampling', type=str, default='greedy', choices=['greedy','random'],help="sampling strategy(greedy and random)")
    parser.add_argument('--eval_sampling', type=str, default='greedy', choices=['greedy','random'],help="evaluation sampling strategy(greedy and random)")
    parser.add_argument('--temp', type=float, choices = ['0.5','1','2'], default='0.5', help="Temperature parameters")
    parser.add_argument('--eval', action='store_true', help='Generate new text') 
    parser.add_argument('--best_m', type=str, default='model_best_greedy.pth.tar', help="Path to best checkpoint")
    parser.add_argument('--text', type=str, default='And now he had to use his own ', help="Path to best checkpoint")
    parser.add_argument('--extnd_length', type=int, default=500, help='Length of an extended sequence')

    config = parser.parse_args()

    # Train the model
    if not config.eval:
        train(config)

    # Bonus: Evaluate
    if config.eval:
        evaluate(config)
