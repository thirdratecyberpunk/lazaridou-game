# class responsible for representing a pair of Agents in the referential game
import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Sigmoid, Module, Linear
import torch.nn.init as Init
from torch.autograd import Variable

class LinearSigmoid(Module):
  def __init__(self, input_dim=32, h_units=32):
      super(LinearSigmoid, self).__init__()
      # sets the initial weights as values from a normal distribution
      w_init = torch.empty(input_dim, h_units).normal_(mean=0.0, std=0.01)
      # defines weights as a new tensor
      self.w = Variable(torch.empty(input_dim, h_units)
      .normal_(mean=0.0, std=0.01), requires_grad=False)
      # sets the biases to contain zeroes
      b_init = torch.zeros(h_units)
      # defines biases as a new tensor
      self.b = Variable(torch.zeros(h_units), requires_grad=False)
      # defines sigmoid function
      self.sig = Sigmoid()

  def forward(self, inputs):
      # returns the result of a Sigmoid function provided the inputs, weights
      # and the biases
      input = torch.mm(inputs, self.w).add(self.b)
      # return Sigmoid(torch.mm(inputs, self.w) + self.b)
      return self.sig(input)

# class representing the model that generates probabilities for each word
class WordProbabilityModel():
    def __init__(self, image_embedding_dim):
        super(WordProbabilityModel, self).__init__()
        # dense network implementation
        self.hidden_layer = Linear(2, 2* image_embedding_dim)
        self.output_layer = Linear(2* image_embedding_dim, 2)

    def forward(x):
        return torch.softmax(super(WordProbabilityModel, self).forward(x), dim=-1)

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
        self.build_sender_receiver_model()
        self.word_probs_model = WordProbabilityModel(self.image_embedding_dim)
        # self.sender_optimizer = Adam(self.sender.parameters(), lr=self.learning_rate)
        # self.receiver_optimizer = Adam(self.receiver.parameters(), lr=self.learning_rate)
        w_init = torch.empty(32, 32).normal_(mean=0.0, std=0.01)
        self.vocab_embedding = Variable(w_init)
        # self.vocab_embedding = Variable(w_init(shape=(self.vocab_len, self.word_embedding_dim), dtype='float32'), trainable=True)

   # TODO: move this logic into a receiver agent class
   # function that returns the image chosen by the receiver agent and the
   # probability distribution for each word
    def get_receiver_selection(self, word, im1_acts, im2_acts):
        word_embed_gather = tf.gather(self.vocab_embedding, self.word)
        # strips any dimensions equal to 1
        word_embed = tf.squeeze(word_embedding_gather)
        # embeds the images into game-specific spaces using the receiver
        im1_embed = self.receiver(im1_acts)
        im2_embed = self.receiver(im2_acts)
        # calculates the score for each image by multiplying the image embed
        # by the word embedding
        im1_score = tf.reduce_sum(tf.multiply(im1_embed, word_embed), axis=1).numpy()[0]
        im2_score = tf.reduce_sum(tf.multiply(im2_embed, word_embed), axis=1).numpy()[0]
        # turns embeddings into probability distribution using the softmax
        # function
        image_probs = tf.nn.softmax([im1_score, im2_score]).numpy()
        # chooses
        selection = np.random.choice(np.arange(2), p=image_probs)
        # returns the probability distribution and chosen image
        return image_probs, selection

    # builds the linear sigmoid architecture for senders/receivers
    # TODO: implement informed/agnostic architectures
    # TODO: implement sender/receiver as their own classes rather than have
    # an Agents class
    def build_sender_receiver_model(self):
        self.sender = LinearSigmoid(1000, self.image_embedding_dim)
        self.receiver = LinearSigmoid(1000, self.image_embedding_dim)

    # TODO: move this logic into a sender agent class
    # function that returns the word chosen by the sender and the probability
    # distribution
    def get_sender_word_probs(self, target_acts, distractor_acts):
        # embeds the target into a game specific embedding space
        t_embed = self.sender(target_acts)
        # embeds the distractor into a game specific embedding space
        d_embed = self.sender(distractor_acts)
        # concatenates embeds into one dimension
        ordered_embed = np.concatenate([t_embed, d_embed], axis=1)
        print(ordered_embed)
        # obtains probability distribution for all words from word prob model
        self.word_probs = self.word_probs_model.forward(ordered_embed).numpy()[0]
        # chooses a word to send by sampling from the probability distribution
        self.word = np.random.choice(np.arange(len(self.vocab)), p=self.word_probs)
        # returns the chosen word and the probability distribution
        return self.word_probs, self.word

    #TODO: is this handled here in a PyTorch workflow?
    def update(self, batch):
        acts, target_acts, distractor_acts, word_probs, \
            receiver_probs, target, word, selection, reward = map(lambda x: np.squeeze(np.array(x)), zip(*batch))
        # normalises variables
        reward = np.reshape(reward, [-1, 1])
        selection = np.reshape(selection, [1, -1])
        word = np.reshape(word, [1, -1])
        target_acts = np.reshape(target_acts, [-1, 1000])
        distractor_acts = np.reshape(distractor_acts, [-1, 1000])
        acts = np.reshape(acts, [-1, 2000])
        receiver_probs = np.reshape(receiver_probs, [-1, 2])
        # calculates loss for sender/receiver
        sender_loss = np.mean(-1 * np.multiply(np.transpose(np.log(selected_word_prob)), reward))
        receiver_loss = np.mean(-1 * np.log(selected_image_prob) * self.reward)
        # updates gradients for optimiser based on loss (gradient descent)
        # TODO: move gradient descent into Game loop

        # with tf.GradientTape() as tape:
        #     sender_gradients = tape.gradients(sender_loss, self.sender.trainable_variables)
        #     self.sender_optimizer.apply_gradients(sender_gradients, self.sender.trainable_variables)
        #
        #     receiver_gradients = tape.gradients(receiver_loss, self.receiver.trainable_variables)
        #     self.receiver_optimizer.apply_gradients(receiver_gradients, self.receiver.trainable_variables)


if __name__=='__main__':

    vocab = ['dog', 'cat', 'mouse']
    agent = Agents(vocab=vocab, image_embedding_dim=10, word_embedding_dim=10,
                   learning_rate=0.2, temperature=10, batch_size=32)
    t_acts = torch.ones((1, 1000))
    d_acts = torch.ones((1, 1000))

    word_probs = agent.get_sender_word(t_acts, d_acts)
