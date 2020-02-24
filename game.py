# class responsible for running a sequence of the game
import sys
import os
import numpy as np
from architectures.senders.AgnosticSender import AgnosticSender
from architectures.receivers.Receiver import Receiver
from architectures.senders.RandomSender import RandomSender
from architectures.receivers.RandomReceiver import RandomReceiver
from architectures.senders.PerfectSender import PerfectSender
import random
import torch
import sys
import argparse
import yaml
import torchvision.models as models
from collections import deque, namedtuple
from torch.optim import Adam
from torch.autograd import Variable
from LossPolicy import LossPolicy
from torch.nn import Embedding
from display import display_comm_succ_graph
from IterableRoundsDataset import IterableRoundsDataset
from torch.utils.data import Dataset, DataLoader

def shuffle_image_activations(im_acts):
    reordering = np.array(range(len(im_acts)))
    random.shuffle(reordering)
    target_ind = np.argmin(reordering)
    shuffled = im_acts[reordering]
    return (shuffled, target_ind)

def run_game(config, device):
    # dimensionality of the image embedding
    image_embedding_dim = config['image_embedding_dim']
    # dimensionality of the word embedding
    word_embedding_dim = config['word_embedding_dim']
    # directory of images
    data_dir = config['data_dir']
    # list of all possible classes to load in
    image_dirs = config['image_dirs']
    # all possible words in the vocabulary of the agent
    vocab = config['vocab']
    # model weights to be passed to the agent
    model_weights = config['model_weights']
    # rate at which the agent learns
    learning_rate = config['learning_rate']
    # number of iterations to run the game for
    iterations = config['iterations']
    # size of batch within batch to sample
    mini_batch_size = config['mini_batch_size']
    # size of batch
    batch_size = config['batch_size']
    # temperature
    temperature = config['temperature']
    # whether the model should be loaded from a pretrained model
    load_model = config['load_model'] == 'True'
    # weight decay -> factor that parameters are multiplied by during loss
    weight_decay = config['weight_decay']
    # whether round configuration should be displayed as a matplotlib diagram
    display_rounds = config['display_rounds']
    # whether results should be displayed as a matplotlib diagram
    display_comm_succ = config['display_comm_succ']
    # number of words in the vocabulary
    vocab_len = len(vocab)

    # creates "perfect" sender and a normal receiver to test whether the receiver is learning at all
    # perfect_sender = PerfectSender(vocab = vocab, input_dim = 1000, h_units= image_embedding_dim, image_embedding_dim= image_embedding_dim)
    receiver = Receiver(1000, image_embedding_dim, image_embedding_dim= image_embedding_dim, word_embedding_dim= word_embedding_dim)

    # defines loss policy for agents
    receiver_loss = LossPolicy()
    # defines optimisers
    receiver_optimizer = Adam(receiver.parameters(), lr=learning_rate, weight_decay = weight_decay)

    # dataset
    dataset = IterableRoundsDataset(image_dirs, data_dir, batch_size = 2)

    # TODO: load in testing data, add testing phase
    # train_data, test_data = train_test_split(dataset, test_size=0.1)
    train_loader = DataLoader(dataset)
    # loads the pretrained VGG16 model
    model = models.vgg16()
    # creates a batch to store all game rounds
    # mathematical definition of game as explained by Lazaridou
    Game = namedtuple("Game", ["im_acts", "target_acts", "distractor_acts",
    "word_probs", "image_probs", "target", "word", "selection", "reward", "selected_word_prob", "selected_image_prob"])

    batch = []
    total_reward = 0
    # total_random_reward = 0
    successes = 0
    # random_successes = 0
    comm_succ = []
    # random_comm_succ = []

    with torch.no_grad():

        # training process
        # sender.train()
        receiver.train()

        # number of training iterations
        for i in range(iterations):
            print("Round {}/{}".format(i, iterations), end = "\n")
            # number of items in the batch
            for index, item in zip(range(mini_batch_size), train_loader):
                # gets a new target/distractor pair from the data set
                target_image, distractor_image = item[0]["arr"], item[1]["arr"]
                # category information for the perfect sender
                target_category, distractor_category = item[0]["category"], item[1]["category"]
                # reshapes images into expected shape for VGG model
                target_image = target_image.reshape((1, 3, 224, 224))
                distractor_image = distractor_image.reshape((1, 3, 224, 224))
                # vertically stacks numpy array of image
                td_images = np.vstack([target_image, distractor_image])
                # gets actual classifications from prediction of vgg model
                td_images_tensor = torch.from_numpy(td_images)
                td_acts = model(td_images_tensor)
                # reshapes predictions into expected shape
                target_acts = td_acts[0].reshape((1, 1000))
                distractor_acts = td_acts[1].reshape((1, 1000))

                # gets the perfect sender's chosen word
                # word_probs, word_selected, selected_word_prob = perfect_sender.forward(
                # target_acts, distractor_acts, target_category)

                # chooses the target's category (perfect play)
                word = target_category
                # as this agent has perfect play, probability is 1 to 0
                if word == 0:
                    word_probs = [1,0]
                else:
                    word_probs = [0,1]
                word_selected = target_category

                print("Perfect sender ssent {}".format(vocab[word_selected]))

                # gets the target image
                # TODO: check if this can be modified for more than 2 images
                reordering = np.array([0,1])
                random.shuffle(reordering)
                target = np.where(reordering==0)[0]
                # sets images as predictions
                img_array = [target_acts, distractor_acts]
                im1_acts, im2_acts = [img_array[reordering[i]]
                for i, img in enumerate(img_array)]
                # gets the receiver's chosen target and associated probability
                receiver_probs, image_selected, selected_image_prob = receiver.forward(
                im1_acts, im2_acts, word_selected)

                print("Receiver chose image {} with a chance of {}, target was image {}".format(image_selected, selected_image_prob, target))
                # print("Random Receiver chose image {} with a chance of {}, target was image {}".format(random_image_selected, random_selected_image_prob, target))
                # gives a payoff if the target is the same as the selected image
                reward = 0.0
                if target == image_selected:
                    reward = 1.0
                    successes += 1
                    print("Success! Payoff of {}".format(reward))

                shuffled_acts = np.concatenate([im1_acts, im2_acts])

                # adds the game just played to the batch
                # batch.append(Game(shuffled_acts, target_acts, distractor_acts,
                # word_probs, receiver_probs, target, word_selected, image_selected,
                # reward, selected_word_prob, selected_image_prob))

                #TO DO: reimplement mini batch descent
                # stochastic update, no batch
                current_comm_succ = successes / (i + 1) * 100
                # random_current_comm_succ = random_successes / (i + 1) * 100
                comm_succ.append(current_comm_succ)
                # random_comm_succ.append(random_current_comm_succ)
                print('Total communication success : {}%'.format(current_comm_succ))
                # print('Total random communication success : {}%'.format(random_current_comm_succ))
                print('Updating the agent weights')
                # sender_optimizer.zero_grad()
                receiver_optimizer.zero_grad()
                total_reward += 1

                # calculates loss for agents
                receiver_loss_value = receiver_loss(selected_image_prob, reward)
                print("Receiver loss {}".format(receiver_loss_value))
                receiver_loss_value.backward()

                # applies gradient descent backwards
                receiver_optimizer.step()

        if (display_comm_succ):
            display_comm_succ_graph({"perfect sender, default receiver":comm_succ})

def main():

    parser = argparse.ArgumentParser(description="Train agents to converge upon a language via a referential game.")
    parser.add_argument('--seed', type = int, default = 0, help="Value used as the seed for random generators")
    parser.add_argument('--conf', required=True, help="Location of configuration file for game.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args = parser.parse_args()
    conf = args.conf

    # Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(conf) as g:
        config = yaml.load(g, Loader=yaml.FullLoader)

    run_game(config,device)

if __name__ == '__main__':
    main()
