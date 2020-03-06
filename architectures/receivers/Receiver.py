import os
import sys
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss

# agent that receives a single word in the vocabulary from the sender
# sends a single image embedding which it thinks is the target image
class Receiver(nn.Module):
    def __init__(self, img_embed_dim, game_embed_dim, word_dict_dim):
        super(Receiver, self).__init__()
        self.img_linear = nn.Linear(img_embed_dim, game_embed_dim)
        self.word_linear = nn.Linear(word_dict_dim, game_embed_dim)
        self.softmax = nn.Softmax()
        self.word_dict_dim = word_dict_dim

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
        selection = np.random.choice(np.arange(self.word_dict_dim), p=prob_distribution)
        return scores, prob_distribution, selection
