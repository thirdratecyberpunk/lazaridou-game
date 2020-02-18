import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Sigmoid, Module, Linear, Sequential, ReLU, Embedding
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss

# class representing the model that generates probabilities for each word
class WordProbabilityModel(Module):
    def __init__(self, image_embedding_dim):
        super(WordProbabilityModel, self).__init__()
        # dense network implementation
        self.s = Sequential(
            Linear(2, 2 * image_embedding_dim),
            ReLU(),
            Linear(2 * image_embedding_dim, 2)
        )

    def forward(self, x):
        # passes the input through the linear layers
        x = self.s(x)
        return torch.softmax(x, dim = 1)

# sender agent that has "perfect" play, i.e. will always send the same
# word for a given image
class PerfectSender(Module):
  def __init__(self, vocab, input_dim=32, h_units=32, image_embedding_dim=2, word_embedding_dim=2):
      super(PerfectSender, self).__init__()
      self.vocab = vocab
      # has a single linear layer to embed the images
      self.linear1 = Linear(input_dim, h_units, bias=None)
      # defines weights as a new tensor
      self.linear1.weight = torch.nn.Parameter(torch.empty(input_dim, h_units).normal_(mean=0.0, std=0.01), requires_grad=True)
      # defines sigmoid function
      self.sig = Sigmoid()

      self.add_module("word_prediction_model", WordProbabilityModel(image_embedding_dim))
      # embedding layer for images into game-specific space
      self.add_module("image_embedding", Embedding(input_dim, image_embedding_dim))
      # embedding layer for vocabulary
      self.add_module("vocab_embedding", Embedding(input_dim, word_embedding_dim))

  def embed_image_to_gss(self, inputs):
      """
      Embeds a given image representation into a game specific space
      """
      input = torch.matmul(inputs, self.linear1.weight)
      embed = self.sig(input)
      return embed

  def forward(self, target, distractor, target_category):
      # chooses the target's category (perfect play)
      word = target_category
      # as this agent has perfect play, probability is 1 to 0
      if word == 0:
          word_probs = [1,0]
      else:
          word_probs = [0,1]
      # samples the word from the vocabulary embedding
      word_embedding = self.vocab_embedding(torch.tensor(word))
      # returns the chosen word, the probability distribution and the
      # probability of choosing that word
      return word_probs, word, word_embedding, word_probs[word]
