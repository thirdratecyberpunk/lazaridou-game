import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Sigmoid, Softmax, Module, Linear
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss

# agent that receives a single word in the vocabulary from the sender
# sends a single image embedding which it thinks is the target image
class Receiver(Module):
  def __init__(self, input_dim=32, h_units=32):
      super(Receiver, self).__init__()
      # has a single layer to embed the images
      self.linear1 = Linear(input_dim, h_units, bias=None)
      # defines weights as a new tensor
      self.linear1.weight = torch.nn.Parameter(torch.empty(input_dim, h_units).normal_(mean=0.0, std=0.01), requires_grad=True)
      # sets the biases to contain zeroes
      b_init = torch.zeros(h_units)
      # defines biases as a new tensor
      self.b = torch.nn.Parameter(torch.zeros(h_units), requires_grad=True)
      # defines sigmoid function
      self.sig = Sigmoid()
      # defines softmax function
      self.softmax = Softmax()

  def embed_image_to_gss(self, inputs):
       """
       Embeds a given image representation into a game specific space
       """
       input = torch.mm(inputs, self.linear1.weight).add(self.b)
       embed = self.sig(input)
       return embed

  def forward(self, image_1, image_2, word_embed):
      # embeds images into game specific space
      im1_embed = Receiver.embed_image_to_gss(self, image_1)
      im2_embed = Receiver.embed_image_to_gss(self, image_2)
      # embeds symbol given as vector into game specific space
      # word_embed = torch.squeeze(vocab_embedding[word])
      # print(word)
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
