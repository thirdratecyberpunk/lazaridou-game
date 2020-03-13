import os
import sys
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss
import sys

# agent that takes in two images and outputs a word choice
class AgnosticSender(nn.Module):
    def __init__(self, img_embed_dim, game_embed_dim, word_dict_dim):
        super(AgnosticSender, self).__init__()
        # layer responsible for embedding the images into game specific space
        self.img_linear = nn.Linear(img_embed_dim, game_embed_dim)
        # used to convert ranges into probabilities
        self.softmax = nn.Softmax()
        # layer responsible for turning an embedding concatenation into a range of scores
        self.embed_linear = nn.Linear(game_embed_dim, 1) 
        # sigmoid non-linearity applied to image embeddings
        self.sigmoid = nn.Sigmoid()
        # size of the vocabulary
        self.word_dict_dim = word_dict_dim

    def forward(self, target, distractor):
        """
        Chooses a word by calculating scores
        Takes the target and distractor vectors
        """
        # embeds images in game specific space
        target_emb = self.img_linear(target)
        distractor_emb = self.img_linear(distractor)
        # applies sigmoid non-linearity to embeddings
        target_sig = self.sigmoid(target_emb)
        distractor_sig = self.sigmoid(distractor_emb)
        # generates scores for each vocabulary symbol by applying weights 
        target_score = self.embed_linear(target_sig)
        distractor_score = self.embed_linear(distractor_sig)
        scores = torch.tensor([target_score, distractor_score], requires_grad=True)
        scores_no_grad = scores.clone().detach()
        # generates a probability distribution from the scores
        prob_distribution = self.softmax(scores_no_grad).cpu().numpy()
        # converts dot products into Gibbs distribution
        # choose word symbol by sampling from Gibbs distribution
        selection = np.random.choice(np.arange(self.word_dict_dim), p=prob_distribution)
        return scores, prob_distribution, selection
