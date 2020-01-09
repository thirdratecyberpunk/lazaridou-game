'''
Running the game loop.
'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as nnfunc
import torch.optim as optim
from PIL import Image
from TargetsDataset import TargetsDataset
from sklearn.model_selection import train_test_split
import random
import time
import csv
import os


parser= argparse.ArgumentParser(description='Train agents to converge upon an agreed language.')
parser.add_argument('--root_dir', default='data/images', help="Root directory of class images.")
parser.add_argument('--seed', type=int, default=0, help="Value used as the seed for random values.")
parser.add_argument('--epochs', type=int, default=5, help="Amount of generations to train the model for.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# Random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# dataset
dataset = TargetsDataset(data_dir=args.root_dir, transform = transforms.Compose(
        [
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.ToTensor()
        ]))

train_data, test_data = train_test_split(dataset, test_size=0.2)

train_loader = DataLoader(train_data, batch_size = 5, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size = 5, shuffle=True, num_workers=2)

# model
# net = inception_v3(pretrained=True, aux_logits=False).to(device)
#
# loss_function = nn.CrossEntropyLoss()
# optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# print("Training...")
#
# for epoch in range(args.epochs):
#     print("Starting epoch " + str(epoch))
#     running_loss = 0.0
#     net.train()
#     # training the network
#     for i, data in enumerate(train_loader, 0):
#         # gets inputs
#         image = data.get('img_tensor').to(device)
#         label = data.get('ellipses').to(device)
#         # zero parameter gradients
#         optimiser.zero_grad()
#         # forward, back and optimise
#         outputs = net(image)
#         loss = loss_function(outputs, label)
#         loss.backward()
#         optimiser.step()
#         # print statistical information
#         running_loss += loss.item()
#
#     net.eval()
#     # evaluating performance at this epoch
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             images = data.get('img_tensor').to(device)
#             labels = data.get('ellipses').to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100.00 * correct / total
#
# print("Finished training")
