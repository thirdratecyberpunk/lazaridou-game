#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:33:37 2020

@author: vogiatzg
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



random.seed(0)

img_dirs = ["dog", "cat"]
data_dir = "data"

iterable_dataset = IterableRoundsDataset(img_dirs, data_dir, batch_size = 2)

loader = DataLoader(iterable_dataset, batch_size = None)

            
            
img_embed_dim = 1000
word_dict_dim = 2

game_embed_dim = 32


model = models.vgg16()

class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()
        self.img_linear = nn.Linear(img_embed_dim, game_embed_dim)
        self.word_linear = nn.Linear(word_dict_dim, game_embed_dim)
        
    def forward(self, im1, im2, w):
        im1_emb = self.img_linear(im1)
        im2_emb = self.img_linear(im2)
        w_emb = self.word_linear(w)
        im_emb = torch.stack((im1_emb,im2_emb),dim=2)
        return torch.einsum('ij,ijk->ik', w_emb, im_emb)
        

receiver = Receiver()

loss = nn.CrossEntropyLoss()

optim_SGD = optim.SGD(receiver.parameters(), lr=0.0001, momentum=0.5)

for batch in islice(loader, 5):
        # reshapes the image tensor into the expected shape
        target_display = model(batch[0]["arr"][None,:,:,:])
        distractor_display = model(batch[1]["arr"][None,:,:,:])
        
        
        w = torch.eye(2)[batch[0]['category']][None,:]
                
        if random.random()<0.5:
            im1 = target_display
            im2 = distractor_display
            t = 0
        else:            
            im2 = target_display
            im1 = distractor_display
            t = 1
            
        optim_SGD.zero_grad()
        L = loss(receiver(im1,im2,w), torch.tensor([t]))
        print(L)
        L.backward()
        optim_SGD.step()

    

