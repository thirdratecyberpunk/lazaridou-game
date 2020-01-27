import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Sigmoid, Module, Linear
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss

# class representing the model that generates probabilities for each word
class WordProbabilityModel():
    def __init__(self, image_embedding_dim):
        super(WordProbabilityModel, self).__init__()
        # dense network implementation
        self.hidden_layer = Linear(2, 2 * image_embedding_dim)
        self.output_layer = Linear(2 * image_embedding_dim, 2)

    def forward(self, x):
        # passes the input through the linear layers
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return torch.softmax(x, dim = 1)

# agent that takes in the target, any distractors and the vocabulary
# and sends a single word from the vocabulary
class AgnosticSender(Module):
  # def __init__(self, image_embedding_dim, input_dim=32, h_units=32):
  def __init__(self, vocab, input_dim=32, h_units=32, image_embedding_dim=2):
      super(AgnosticSender, self).__init__()
      print(input_dim, h_units)
      self.vocab = vocab
      # has a single layer to embed the images
      self.linear1 = Linear(input_dim, h_units, bias=None)
      # setting the weights for the linear layer as a parameter
      w_init = torch.empty(input_dim, h_units).normal_(mean=0.0, std=0.01)
      # defines weights as a new tensor
      self.linear1.weight = torch.nn.Parameter(torch.empty(input_dim, h_units)
      .normal_(mean=0.0, std=0.01), requires_grad=True)
      self.w = torch.nn.Parameter(torch.empty(input_dim, h_units)
      .normal_(mean=0.0, std=0.01), requires_grad=True)

      # sets the biases to contain zeroes
      b_init = torch.zeros(h_units)
      # defines biases as a new tensor
      self.b = torch.nn.Parameter(torch.zeros(h_units), requires_grad=True)
      # defines sigmoid function
      self.sig = Sigmoid()

      self.word_prediction_model = WordProbabilityModel(image_embedding_dim)

  def embed_image_to_gss(self, inputs):
      """
      Embeds a given image representation into a game specific space
      """
      input = torch.mm(inputs, self.linear1.weight).add(self.b)
      # input = self.linear1(inputs)
      embed = self.sig(input)
      # print(embed)
      return embed


  def forward(self, target, distractor):
      # embeds target in game embedding space
      target_embedding = AgnosticSender.embed_image_to_gss(self, target)
      # embeds distraction in game embedding space
      distractor_embedding = AgnosticSender.embed_image_to_gss(self, distractor)
      # concatenates embeddings
      ordered_embed = np.concatenate([target_embedding, distractor_embedding], axis=0)
      ordered_embed_tensor = torch.from_numpy(ordered_embed)
      # obtains probability distribution for all words
      word_probs = self.word_prediction_model.forward(ordered_embed_tensor).numpy()[0]
      # chooses a word to send by sampling from the probability distribution
      word = np.random.choice(np.arange(len(self.vocab)), p=word_probs)
      # returns the chosen word, the probability distribution and the
      # probability of choosing that word
      return word_probs, word, word_probs[word]
