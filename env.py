# class responsible for representing the referential game itself
# Environment contains information about the possible words,
# target + distractor images and score for the game
from torchvision import transforms, utils
import numpy
from PIL import Image
import skimage
from skimage import io
import matplotlib.pyplot as plt

# loads an image from a directory and applies a transformation to it
def load_image(path):
    image = Image.open(path)
    # transformation rescales + randomly crops the image to 32*32
    transform = transforms.Compose(
            [
            transforms.Resize((128,128)),
            transforms.RandomCrop(32)
            ])
    plt.figure()
    image = transform(image)
    arr = numpy.array(image)
    plt.imshow(arr)
    plt.show()

class Environment:
    def __init__(self, data_dir, img_dirs, num_classes):
        # directory of images for classes
        self.img_dirs = img_dirs
        # directory of data
        self.data_dir = data_dir
        # number of possible classes
        self.num_classes = num_classes
        # word sent by the sender agent
        self.word = None
        # target image for a round of a game
        self.target = None
        # distractor image for a round of a game
        # TODO: make this an array, experiment with multiple distractors
        self.distractor = None
        # class of the target image
        self.target_class = None

    # updates word sent by the sender agent
    def send_word(self, sent_word):
        self.word = sent_word

    # gets word sent by the sender agent
    def get_word(self):
        return self.word

    # resets the information from agents
    def reset(self):
        self.target = None
        self.distractor = None
        self.word = None

load_image("data/images/cat/0.jpg")
