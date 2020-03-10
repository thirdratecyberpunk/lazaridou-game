#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:33:37 2020

@author: vogiatzg
@author: blackbul
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchvision import transforms, utils
import torchvision.models as models
from PIL import Image
import random
from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import islice, chain, cycle
import os
import numpy as np
from display import plot_figures
from itertools import repeat

import argparse

from display import plot_figures, display_comm_succ_graph, display_loss_graph
from IterableRoundsDataset import IterableRoundsDataset
from architectures.receivers.Receiver import Receiver

parser = argparse.ArgumentParser(description="Train agents to converge upon a language via a referential game.")
parser.add_argument('--seed', type = int, default = 0, help="Value used as the seed for random generators")
parser.add_argument('--epochs', type= int, default = 100000, help="Number of epochs to run")
parser.add_argument('--lr', type=float, default = 0.0001, help="Learning rate")
parser.add_argument('--word_dict_dim', type=int, default= 2, help="Size of the vocabulary")

args = parser.parse_args()
seed = args.seed
epochs = args.epochs
lr = args.lr
word_dict_dim = args.word_dict_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

img_dirs = ["cat-1", "mice-1"]
data_dir = "data"

iterable_dataset = IterableRoundsDataset(img_dirs, data_dir, batch_size = 2)

loader = DataLoader(iterable_dataset, batch_size = None)

img_embed_dim = 1000

game_embed_dim = 32

model = models.vgg16().to(device)

receiver = Receiver(img_embed_dim, game_embed_dim, word_dict_dim).to(device)

# TODO: implement the loss function as defined in the paper
# negative expectation of the payoff
receiver_loss = nn.CrossEntropyLoss()

def sender_loss(symbol_prob, payoff):
    return torch.ones(1, requires_grad = True) 

# TODO: try using Adam as the optimizer rather than stochastic gradient descent
receiver_optim_SGD = optim.SGD(receiver.parameters(), lr=lr, momentum=0.5)
# sender_optim_SGD = optim.SGD(sender.parameters(), lr=lr, momentum = 0.5)

total_rounds = 0
total_successes = 0
comm_success_rate = []
receiver_loss_rate = []
sender_loss_rate = []
num_rounds = 1

vocabulary = torch.eye(word_dict_dim).to(device)

for batch in islice(loader, epochs):
        # reshapes the image tensor into the expected shape
        target = model(batch[0]["arr"][None,:,:,:].to(device))
        distractor = model(batch[1]["arr"][None,:,:,:].to(device))
        
        # SENDER LOGIC
        # in this case, "perfect" play is assumed
        # sender will always send the same word for the same category
        # chooses the word as a one hot encoding vector
        w = vocabulary[batch[0]['category']][None,:].to(device)
        sender_scores = [1, 0]
        # learning sender

        # shuffles the targets and distractors so receiver doesn't learn target based on position
        if random.random()<0.5:
            im1 = target
            im1_category = batch[0]["category"]
            im2 = distractor
            im2_category = batch[1]["category"]
            t = 0
        else:            
            im1 = distractor
            im1_category = batch[1]["category"]
            im2 = target
            im2_category = batch[0]["category"]
            t = 1

        total_rounds += 1
        print(f"Round {total_rounds}")
        print(f"Sender sent word {w} for target {t}")


        # RECEIVER LOGIC
        # hardcoded "perfect" receiver
        # receiver will always choose the given image for a given word
        if torch.eq(w, vocabulary[im1_category]).all():
            receiver_choice = 0
        else:
            receiver_choice = 1

        print(f"Receiver chose {receiver_choice}")

        # learning receiver
        # receiver "points" to an image
        # receiver_scores, receiver_prob_distribution, receiver_choice = receiver(im1,im2,w)
        # print(f"Receiver chose {receiver_choice} with a probability of {receiver_prob_distribution[receiver_choice]}")

        payoff = 0
        # checks if the receiver correctly chose the image
        if receiver_choice == t:
            print("Success!")
            total_successes += 1
            payoff = 1
        else:
            print("Failure")
        comm_success_rate.append(total_successes / total_rounds * 100)


        # # applies backpropagation of loss for receiver
        # receiver_optim_SGD.zero_grad()
        # receiver_loss_value = receiver_loss(receiver_scores, torch.tensor([t]).to(device))
        # print(receiver_loss_value)
        # print("_______________________________")
        # receiver_loss_rate.append(receiver_loss_value)
        # receiver_loss_value.backward()
        # receiver_optim_SGD.step()


        # # applies backpropagation of loss for sender
        # sender_optim.SGD.zero_grad()
        # sender_loss_value = sender_loss(sender_scores, payoff)
        # print(sender_loss_value)
        # print("_______________________________")
        # sender_loss_rate.append(sender_loss_value)
        # sender_loss_value.backward()
        # sender_optim_SGD.step()

print(f"{total_successes/total_rounds * 100}% games successful")
display_loss_graph(loss_rate)
# display_comm_succ_graph({"perfect sender, default receiver":comm_success_rate})