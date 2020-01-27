# class responsible for representing a pair of Agents in the referential game
import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Sigmoid, Module, Linear
import torch.nn.init as Init
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss
from AgnosticSender import AgnosticSender

class LossPolicy(Module):
    def __init__(self):
        super(LossPolicy,self).__init__()

    def forward(self, selected_word_prob ,reward):
        result = np.mean(-1 * np.multiply(np.transpose(np.log(selected_word_prob)), reward))
        res_tensor = torch.tensor(result, requires_grad = True)
        return res_tensor

class LinearSigmoid(Module):
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

class Agents:
    # class containing pair of agents:
    # the sender agent, who receives the activations of possible images, maps
    # them to vocabulary words and sends a word to the receiver
    # the receiver agent, who receives the word, maps that word into the
    # embedding and chooses a target image
    def __init__(self, vocab, image_embedding_dim, word_embedding_dim,
                 learning_rate, temperature=10, batch_size=32):
        self.vocab = vocab
        self.image_embedding_dim = image_embedding_dim
        self.batch_size = batch_size
        self.word_embedding_dim = word_embedding_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.vocab_len = len(self.vocab)
        # TODO: move models to device
        self.sender = AgnosticSender(1000, self.image_embedding_dim)
        self.receiver = Receiver(1000, self.image_embedding_dim)
        self.word_probs_model = WordProbabilityModel(self.image_embedding_dim)
        self.sender_optimizer = Adam(self.sender.parameters(), lr=self.learning_rate)
        self.receiver_optimizer = Adam(self.receiver.parameters(), lr=self.learning_rate)
        w_init = torch.empty(self.vocab_len, self.word_embedding_dim).normal_(mean=0.0, std=0.01)
        self.vocab_embedding = Variable(w_init, requires_grad = True)
        # self.vocab_embedding = Variable(w_init(shape=(self.vocab_len, self.word_embedding_dim), dtype='float32'), trainable=True)
        self.loss = LossPolicy()


    # builds the linear sigmoid architecture for senders/receivers
    # TODO: implement informed/agnostic architectures
    # TODO: implement sender/receiver as their own classes rather than have
    # an Agents class
    def build_sender_receiver_model(self):
        self.sender = LinearSigmoid(1000, self.image_embedding_dim)
        self.receiver = LinearSigmoid(1000, self.image_embedding_dim)

   # TODO: move this logic into a receiver agent class
   # function that returns the image chosen by the receiver agent and the
   # probability distribution for each word
    def get_receiver_selection(self, word, im1_acts, im2_acts):
        # slices the vocab embedding using the word as the indice
        word_gathered = self.vocab_embedding[self.word]
        # strips any dimensions equal to 1
        word_embed = torch.squeeze(word_gathered)
        # embeds the images into game-specific spaces using the receiver
        im1_embed = self.receiver(im1_acts)
        im2_embed = self.receiver(im2_acts)
        # calculates the score for each image by multiplying the image embed
        # by the word embedding, then summing along the columns
        im1_mm = torch.mul(im1_embed, word_embed)
        im1_score = torch.sum(im1_mm, dim=1).numpy()[0]

        im2_mm = torch.mul(im2_embed, word_embed)
        im2_score = torch.sum(im2_mm, dim=1).numpy()[0]

        # turns embeddings into probability distribution using the softmax
        # function
        image_probs = torch.softmax(torch.FloatTensor([im1_score, im2_score]), dim=0).numpy()
        # chooses
        selection = np.random.choice(np.arange(2), p=image_probs)
        # returns the probability distribution and chosen image
        return image_probs, selection, image_probs[selection]


    # TODO: move this logic into a sender agent class
    # function that returns the word chosen by the sender and the probability
    # distribution
    def get_sender_word_probs(self, target_acts, distractor_acts):
        # embeds the target into a game specific embedding space
        t_embed = self.sender(target_acts)
        # embeds the distractor into a game specific embedding space
        d_embed = self.sender(distractor_acts)
        # concatenates embeds into one dimension
        ordered_embed = np.concatenate([t_embed, d_embed], axis=0)
        ordered_embed_tensor = torch.from_numpy(ordered_embed)
        # obtains probability distribution for all words from word prob model
        self.word_probs = self.word_probs_model.forward(ordered_embed_tensor).numpy()[0]
        # chooses a word to send by sampling from the probability distribution
        self.word = np.random.choice(np.arange(len(self.vocab)), p=self.word_probs)
        # returns the chosen word, the probability distribution and the
        # probability of choosing that word
        return self.word_probs, self.word, self.word_probs[self.word]

    #TODO: is this handled here in a PyTorch workflow?
    def update(self, batch):
        # obtains these variables from every game in the Batch
        # ACTS is more than 1d
        # acts, target_acts, distractor_
        #sender_loss = np.mean(-1 * np.multiply(np.transpose(np.log(selected_word_prob)), reward))
        #receiver_loss = np.mean(-1 * np.log(selected_image_prob) * reward)acts, word_probs, \
        #     receiver_probs, target, word, selection, reward = map(lambda x: np.squeeze(x.numpy()), zip(*batch))

        zip_batch = list(zip(*batch))
        # TODO: tidy this up
        acts = np.squeeze(np.array(zip_batch[0]))
        target_acts = np.squeeze(np.array(torch.stack(zip_batch[1])))
        distractor_acts = np.squeeze(np.array(torch.stack(zip_batch[2])))
        word_probs = np.squeeze(np.array(zip_batch[3]))
        receiver_probs = np.squeeze(np.array(zip_batch[4]))
        target = np.squeeze(np.array(zip_batch[5]))
        word = np.squeeze(np.array(zip_batch[6]))
        selection = np.squeeze(np.array(zip_batch[7]))
        reward = np.squeeze(np.array(zip_batch[8]))
        selected_word_prob = np.squeeze(np.array(zip_batch[9]))
        selected_image_prob = np.squeeze(np.array(zip_batch[10]))

        # mean_word_probs = np.mean(word_probs).squeeze()

        # normalises variables
        reward = np.reshape(reward, [-1, 1])
        selection = np.reshape(selection, [1, -1])
        word = np.reshape(word, [1, -1])
        target_acts = np.reshape(target_acts, [-1, 1000])
        distractor_acts = np.reshape(distractor_acts, [-1, 1000])
        acts = np.reshape(acts, [-1, 2000])
        receiver_probs = np.reshape(receiver_probs, [-1, 2])
        word_probs = np.reshape(word_probs, [-1, 2])

        # turns a collection of probability distributions for a word
        # i.e. [[0.9,0.1], [0.7,0.3]]
        # into a single average probability distribution
        # i.e. [0.8,0.2]
        word_probs_tensor = torch.tensor(prob_range_to_average_distribution(word_probs))
        word_tensor = torch.tensor(word)

        self.sender.train()
        self.receiver.train()

        # calculates loss for sender/receiver
        # not sure about the cross entropy loss here?
        # sender_loss_policy = CrossEntropyLoss()
        # sender_loss = sender_loss_policy(input=word_probs_tensor,target=word_tensor)
        # sender_loss.backwards()
        sender_loss = torch.tensor(-1 * np.mean(np.multiply(prob_range_to_average_distribution(word_probs), reward)), requires_grad=True)
        sender_loss.backward()

        # receiver_loss_policy = CrossEntropyLoss()
        receiver_loss_policy = NLLLoss()
        selection_list = selection.tolist()[0]
        receiver_probs_tensor = torch.tensor(np.log(receiver_probs), requires_grad = True)
        selection_tensor = torch.tensor(selection_list)
        receiver_loss = torch.tensor(receiver_loss_policy(input=receiver_probs_tensor, target=selection_tensor), requires_grad=True)
        receiver_loss.backward()

        # self.sender_optimizer.zero_grad()
        # self.receiver_optimizer.zero_grad()
        self.sender_optimizer.step()
        self.receiver_optimizer.step()

        self.sender.eval()
        self.receiver.eval()

def prob_range_to_average_distribution(prob_range):
    """
    turns a collection of probability distributions for a word
    i.e. [[0.9,0.1], [0.7,0.3]]
    into a single average log probability distribution
    i.e. [0.8,0.2]
    """
    return np.mean(np.transpose(np.log(prob_range)), axis=1)

if __name__=='__main__':

    vocab = ['dog', 'cat']
    agent = Agents(vocab=vocab, image_embedding_dim=10, word_embedding_dim=10,
                   learning_rate=0.2, temperature=10, batch_size=32)
    t_acts = torch.ones((1, 1000))
    d_acts = torch.ones((1, 1000))

    word_probs = agent.get_sender_word(t_acts, d_acts)
