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

loss = nn.CrossEntropyLoss()

# TODO: try using Adam as the optimizer rather than stochastic gradient descent
optim_SGD = optim.SGD(receiver.parameters(), lr=lr, momentum=0.5)
# optimizer = optim.Adam(receiver.parameters(), lr = 0.001)

total_rounds = 0
total_successes = 0
comm_success_rate = []
loss_rate = []
num_rounds = 1

vocabulary = torch.eye(word_dict_dim)

for batch in islice(loader, epochs):
        # displaying the batch as a diagram
        # target = batch[0]["arr"].permute(1, 2, 0)
        # distractor = batch[1]["arr"].permute(1, 2, 0)

        # figures = [target, distractor]
        # plot_figures(figures, 1, 2)

        # reshapes the image tensor into the expected shape
        target = model(batch[0]["arr"][None,:,:,:].to(device))
        distractor = model(batch[1]["arr"][None,:,:,:].to(device))
        
        # chooses the word as a one hot encoding vector
        w = vocabulary[batch[0]['category']][None,:].to(device)

        # shuffles the targets and distractors so receiver doesn't learn target based on position
        if random.random()<0.5:
            im1 = target
            im2 = distractor
            t = 0
        else:            
            im2 = target
            im1 = distractor
            t = 1

        total_rounds += 1
        print(f"Round {total_rounds}")
        print(f"Sender sent word {w} for target {t}")

        # receiver "points" to an image
        receiver_scores, receiver_prob_distribution, receiver_choice = receiver(im1,im2,w)
        print(f"Receiver chose {receiver_choice} with a probability of {receiver_prob_distribution[receiver_choice]}")

        # checks if the receiver correctly chose the image
        if receiver_choice == t:
            print("Success!")
            total_successes += 1
        else:
            print("Failure")
        comm_success_rate.append(total_successes / total_rounds * 100)
        # applies backpropagation of loss
        optim_SGD.zero_grad()
        # optimizer.zero_grad()
        L = loss(receiver_scores, torch.tensor([t]).to(device))
        print(L)
        print("_______________________________")
        loss_rate.append(L)
        L.backward()
        optim_SGD.step()
        # optimizer.zero_grad()

display_loss_graph(loss_rate)
# display_comm_succ_graph({"perfect sender, default receiver":comm_success_rate})