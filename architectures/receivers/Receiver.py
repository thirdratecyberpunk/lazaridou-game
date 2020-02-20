import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Sigmoid, Softmax, Module, Linear, Embedding
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss

# agent that receives a single word in the vocabulary from the sender
# sends a single image embedding which it thinks is the target image
class Receiver(Module):
  def __init__(self, input_dim=32, h_units=32, word_embedding_dim=2):
      super(Receiver, self).__init__()
      # has a single layer to embed the images
      self.linear1 = Linear(input_dim, h_units)
      # defines weights as a new tensor
      self.linear1.weight = torch.nn.Parameter(torch.empty(input_dim, h_units).normal_(mean=0.0, std=0.01), requires_grad=True)
      # defines sigmoid function
      self.sig = Sigmoid()
      # defines softmax function
      self.softmax = Softmax()
      # embedding layer for vocabulary
      self.add_module("vocab_embedding", Embedding(input_dim, word_embedding_dim))

  def embed_image_to_gss(self, inputs):
       """
       Embeds a given image representation into a game specific space
       """
       input = torch.matmul(inputs, self.linear1.weight)
       embed = self.sig(input)
       return embed

  def forward(self, image_1, image_2, word):
      """
      Chooses an image by calculating scores
      Takes the target and distractor vectors, and the word symbol
      as a one-hot vector over the vocabulary
      """
      # embeds images into game specific space
      im1_embed = Receiver.embed_image_to_gss(self, image_1)
      im2_embed = Receiver.embed_image_to_gss(self, image_2)
      # embeds symbol given as vector into game specific space
      word_embed = self.vocab_embedding(word)
      # word_embed = Receiver.embed_image_to_gss(self, word)
      # computes dot product between symbol and images
      im1_mm = torch.mul(im1_embed, word_embed)
      im1_score = torch.sum(im1_mm, dim=1).numpy()[0]

      im2_mm = torch.mul(im2_embed, word_embed)
      im2_score = torch.sum(im2_mm, dim=1).numpy()[0]

      scores = torch.FloatTensor([im1_score, im2_score])
      # converts dot products into Gibbs distribution
      image_probs = self.softmax(scores).numpy()
      # choose image by sampling from Gibbs distribution
      selection = np.random.choice(np.arange(2), p=image_probs)
      # return the probability distribution, chosen word, and probability of chosen word
      return image_probs, selection, image_probs[selection]
