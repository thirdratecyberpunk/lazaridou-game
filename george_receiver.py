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
from IterableRoundsDataset import IterableRoundsDataset


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

from display import plot_figures, display_comm_succ_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dtype = torch.cuda.FloatTensor

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

img_dirs = ["cat-1", "dog-1"]
data_dir = "data"

iterable_dataset = IterableRoundsDataset(img_dirs, data_dir, batch_size = 2)

loader = DataLoader(iterable_dataset, batch_size = None)

img_embed_dim = 1000
word_dict_dim = 2

game_embed_dim = 32


model = models.vgg16().to(device)

class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()
        self.img_linear = nn.Linear(img_embed_dim, game_embed_dim)
        self.word_linear = nn.Linear(word_dict_dim, game_embed_dim)
        self.softmax = nn.Softmax()

    def forward(self, im1, im2, w):
        """
        Chooses an image by calculating scores
        Takes the target and distractor vectors, and the word symbol
        as a one-hot vector over the vocabulary
        """
        im1_emb = self.img_linear(im1)
        im2_emb = self.img_linear(im2)
        w_emb = self.word_linear(w)
        im_emb = torch.stack((im1_emb,im2_emb),dim=2)
        scores = torch.einsum('ij,ijk->ik', w_emb, im_emb)
        scores_no_grad = scores.clone().detach()
        #   converts dot products into Gibbs distribution
        prob_distribution = self.softmax(scores_no_grad).cpu().numpy()[0]
        # choose image by sampling from Gibbs distribution
        selection = np.random.choice(np.arange(2), p=prob_distribution)
        return scores, prob_distribution, selection
        

receiver = Receiver().to(device)

loss = nn.CrossEntropyLoss()

optim_SGD = optim.SGD(receiver.parameters(), lr=0.0001, momentum=0.5)
# optimizer = optim.Adam(receiver.parameters(), lr = 0.001)

total_rounds = 0
total_successes = 0
comm_success_rate = []
num_rounds = 1

for batch in islice(loader, 100000):
        # displaying the batch as a diagram
        # target_display = batch[0]["arr"].permute(1, 2, 0)
        # distractor_display = batch[1]["arr"].permute(1, 2, 0)

        # figures = [target_display, distractor_display]
        # plot_figures(figures, 1, 2)

        # reshapes the image tensor into the expected shape
        target_display = model(batch[0]["arr"][None,:,:,:].to(device))
        distractor_display = model(batch[1]["arr"][None,:,:,:].to(device))
        
        # chooses the word as a one hot encoding vector
        w = torch.eye(2)[batch[0]['category']][None,:].to(device)

        # shuffles the targets and distractors so receiver doesn't learn target based on position
        if random.random()<0.5:
            im1 = target_display
            im2 = distractor_display
            t = 0
        else:            
            im2 = target_display
            im1 = distractor_display
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
        L.backward()
        optim_SGD.step()
        # optimizer.step()

display_comm_succ_graph({"perfect sender, default receiver":comm_success_rate})

