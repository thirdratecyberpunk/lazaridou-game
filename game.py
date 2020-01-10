# class responsible for running a sequence of the game
import sys
import os
import numpy as np
from agent import Agents
import env
import random
import torch
import sys
import argparse
import yaml
import torchvision.models as models
from collections import deque, namedtuple


def shuffle_image_activations(im_acts):
    reordering = np.array(range(len(im_acts)))
    random.shuffle(reordering)
    target_ind = np.argmin(reordering)
    shuffled = im_acts[reordering]
    return (shuffled, target_ind)

def run_game(config):
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

    # creates a pair of sender/receiver agents which are used to populate the
    # game TODO: replace this with a separate sender/receiver class to experiment
    # with different population numbers
    agents = Agents(vocab,
                    image_embedding_dim,
                    word_embedding_dim,
                    learning_rate,
                    temperature,
                    batch_size)

    # creates a referential game environment
    # TODO: modify the environment so it can take more classes/distractors
    environ = env.Environment(data_dir, image_dirs, 2)

    total_rounds = 0
    wins = 0
    losses = 0

    # loads the pretrained VGG16 model
    model = models.vgg16()
    # creates a batch to store all game rounds
    batch = []
    # mathematical definition of game as explained by Lazaridou
    Game = namedtuple("Game", ["im_acts", "target_acts", "distractor_acts",
    "word_probs", "image_probs", "target", "word", "selection", "reward"])

    total_reward = 0
    successes = 0
    with torch.no_grad():
        for i in range(iterations):
            print("Round {}/{}".format(i, iterations), end = "\n")
            # gets a new target/distractor pair from the environment
            target_image, distractor_image = environ.get_images()
            # TODO: check if this is needed
            # reshapes images into expected shape for VGG model
            target_image = target_image.reshape((1, 3, 224, 224))
            distractor_image = distractor_image.reshape((1, 3,224, 224))
            # sets the target class variable
            target_class = environ.target_class
            # vertically stacks numpy array of image
            td_images = np.vstack([target_image, distractor_image])

            # gets actual classifications from prediction of vgg model
            td_images_tensor = torch.from_numpy(td_images)
            td_acts = model(td_images_tensor)
            # reshapes predictions
            target_acts = td_acts[0].reshape((1, 1000))
            distractor_acts = td_acts[1].reshape((1, 1000))
            # gets the sender's chosen word and the associated probability
            word_probs, word_selected = agents.get_sender_word_probs(
            target_acts, distractor_acts)
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
            receiver_probs, image_selected = agents.get_receiver_selection(
            word_selected, im1_acts, im2_acts)
            # gives a payoff if the target is the same as the selected image
            reward = 0.0
            if target == image_selected:
                reward = 1.0
                successes += 1
            shuffled_acts = np.concatenate([im1_acts, im2_acts])
            # adds the game just played to the batch
            batch.append(Game(shuffled_acts, target_acts, distractor_acts,
            word_probs, receiver_probs, target, word_selected, image_selected,
            reward))

            #TODO: implement weight updates
            if (i+1) % mini_batch_size == 0:
                print('updating the agent weights')
                accuracy = successes / i * 100
                print('Accuracy : {}%'.format(accuracy))

            print('Target {}, chosen {}, reward {}'.format(target, image_selected, reward))
            total_reward += reward

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True)
    args = parser.parse_args()
    conf = args.conf

    with open(conf) as g:
        config = yaml.load(g, Loader=yaml.FullLoader)

    run_game(config)

if __name__ == '__main__':
    main()
