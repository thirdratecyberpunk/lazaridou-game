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
from Receiver import Receiver

class LossPolicy(Module):
    def __init__(self):
        super(LossPolicy,self).__init__()

    def forward(self, selected_word_prob ,reward):
        result = np.mean(np.multiply(np.transpose(np.log(selected_word_prob)), reward))
        res_tensor = torch.tensor(result, requires_grad = True)
        return res_tensor

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
        self.sender = AgnosticSender(vocab = self.vocab, input_dim = 1000, h_units=self.image_embedding_dim, image_embedding_dim=self.image_embedding_dim)
        self.receiver = Receiver(1000, self.image_embedding_dim)
        for n in self.sender.parameters():
            print (n)
        for n in self.receiver.parameters():
            print(n)
        self.sender_optimizer = Adam(self.sender.parameters(), lr=self.learning_rate)
        self.receiver_optimizer = Adam(self.receiver.parameters(), lr=self.learning_rate)
        w_init = torch.empty(self.vocab_len, self.word_embedding_dim).normal_(mean=0.0, std=0.01)
        self.vocab_embedding = Variable(w_init, requires_grad = True)
        self.sender_loss = LossPolicy()
        self.receiver_loss = LossPolicy()
        self.word = None

   # TODO: move this logic into a receiver agent class
   # function that returns the image chosen by the receiver agent and the
   # probability distribution for each word
    def get_receiver_selection(self, word, im1_acts, im2_acts):
        return self.receiver.forward(im1_acts, im2_acts, self.vocab_embedding, word)

    # function that returns the word chosen by the sender and the probability
    # distribution
    def get_sender_word_probs(self, target_acts, distractor_acts):
        return self.sender.forward(target_acts, distractor_acts)

    #TODO: is this handled here in a PyTorch workflow?
    def update(self, game):
        print(game)
        # obtains these variables from every game in the Batch
        acts = game[0]
        target_acts = game[1]
        distractor_acts = game[2]
        word_probs = game[3]
        receiver_probs = game[4]
        target = game[5]
        word = game[6]
        selection = game[7]
        reward = game[8]
        selected_word_prob = game[9]
        selected_image_prob = game[10]

        # normalises variables
        reward = np.reshape(reward, [-1, 1])
        selection = np.reshape(selection, [1, -1])
        word = np.reshape(word, [1, -1])
        target_acts = np.reshape(target_acts, [-1, 1000])
        distractor_acts = np.reshape(distractor_acts, [-1, 1000])
        acts = np.reshape(acts, [-1, 2000])
        receiver_probs = np.reshape(receiver_probs, [-1, 2])
        word_probs = np.reshape(word_probs, [-1, 2])

        self.sender_optimizer.zero_grad()
        self.receiver_optimizer.zero_grad()

        # calculates loss for agents
        sender_loss_value = self.sender_loss(selected_word_prob, reward)
        sender_loss_value.backward()
        print(sender_loss_value.item())

        # receiver_loss_policy = CrossEntropyLoss()
        receiver_loss_value = self.receiver_loss(selected_image_prob, reward)
        receiver_loss_value.backward()
        print(receiver_loss_value.item())

        # applies gradient descent backwards
        self.sender_optimizer.step()
        self.receiver_optimizer.step()

        #TODO: is this handled here in a PyTorch workflow?
        # def update_batch(self, batch):
        #     # obtains these variables from every game in the Batch
        #     zip_batch = list(zip(*batch))
        #     acts = np.squeeze(np.array(zip_batch[0]))
        #     target_acts = np.squeeze(np.array(torch.stack(zip_batch[1])))
        #     distractor_acts = np.squeeze(np.array(torch.stack(zip_batch[2])))
        #     word_probs = np.squeeze(np.array(zip_batch[3]))
        #     receiver_probs = np.squeeze(np.array(zip_batch[4]))
        #     target = np.squeeze(np.array(zip_batch[5]))
        #     word = np.squeeze(np.array(zip_batch[6]))
        #     selection = np.squeeze(np.array(zip_batch[7]))
        #     reward = np.squeeze(np.array(zip_batch[8]))
        #     selected_word_prob = np.squeeze(np.array(zip_batch[9]))
        #     selected_image_prob = np.squeeze(np.array(zip_batch[10]))
        #
        #     # normalises variables
        #     reward = np.reshape(reward, [-1, 1])
        #     selection = np.reshape(selection, [1, -1])
        #     word = np.reshape(word, [1, -1])
        #     target_acts = np.reshape(target_acts, [-1, 1000])
        #     distractor_acts = np.reshape(distractor_acts, [-1, 1000])
        #     acts = np.reshape(acts, [-1, 2000])
        #     receiver_probs = np.reshape(receiver_probs, [-1, 2])
        #     word_probs = np.reshape(word_probs, [-1, 2])
        #
        #     # turns a collection of probability distributions for a word
        #     # i.e. [[0.9,0.1], [0.7,0.3]]
        #     # into a single average probability distribution
        #     # i.e. [0.8,0.2]
        #     word_probs_tensor = torch.tensor(prob_range_to_average_distribution(word_probs))
        #     word_tensor = torch.tensor(word)
        #
        #     self.sender.train()
        #     self.receiver.train()
        #
        #     # calculates loss for agents
        #     sender_loss_value = sender_loss
        #     sender_loss_value.backward()
        #
        #     # receiver_loss_policy = CrossEntropyLoss()
        #     receiver_loss_policy = NLLLoss()
        #     selection_list = selection.tolist()[0]
        #     receiver_probs_tensor = torch.tensor(np.log(receiver_probs), requires_grad = True)
        #     selection_tensor = torch.tensor(selection_list)
        #     receiver_loss = torch.tensor(receiver_loss_policy(input=receiver_probs_tensor, target=selection_tensor), requires_grad=True)
        #     receiver_loss.backward()
        #
        #     # applies gradient descent backwards
        #     self.sender_optimizer.step()
        #     self.receiver_optimizer.step()

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
