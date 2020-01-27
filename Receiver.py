import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Sigmoid, Module, Linear
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss

# agent that receives a single word in the vocabulary from the sender
# sends a single image embedding which it thinks is the target image
class Receiver(Module):
  def __init__(self, input_dim=32, h_units=32):
      super(LinearSigmoid, self).__init__()
      # sets the initial weights as values from a normal distribution
      w_init = torch.empty(input_dim, h_units).normal_(mean=0.0, std=0.01)
      # defines weights as a new tensor
      self.w = torch.nn.Parameter(torch.empty(input_dim, h_units)
      .normal_(mean=0.0, std=0.01), requires_grad=True)

      # sets the biases to contain zeroes
      b_init = torch.zeros(h_units)
      # defines biases as a new tensor
      self.b = torch.nn.Parameter(torch.zeros(h_units), requires_grad=True)
      # defines sigmoid function
      self.sig = Sigmoid()

  def forward(self, inputs):
      # returns the result of a Sigmoid function provided the inputs, weights
      # and the biases
      input = torch.mm(inputs, self.w).add(self.b)
      result = self.sig(input)
      print (result)
      return result
