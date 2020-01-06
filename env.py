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

def get_all_images(path):
    return [i for i in path if i.endswith('.jpg')]

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

    # TODO: tidy this up, feels a little messy
    # TODO: try replacing this with a data loader class? more PyTorchy
    def get_images(self):
        # picks a random class for target/distractor pair
        im1_class, im2_class = np.random.choice(list(range(self.num_classes)),
        2, replace=False)
        # gets directory for images
        im1_dir, im2_dir = self.img_dirs[im1_class], self.img_dirs[im2_class]

        ## Temp var for sender training, remove later
        self.target_class = im1_dir

        im1_path, im2_path = os.path.join(self.data_dir, 'images', im1_dir),
        os.path.join(self.data_dir, 'images', im2_dir)

        # select random image in dirs
        im1_files, im2_files = os.listdir(im1_path), os.listdir(im2_path)
        im1_files = get_all_images(im1_files)
        im2_files = get_all_images(im2_files)

        # selects a random image for each class
        im1 = np.random.choice(len(im1_files), 1)[0]
        im2 = np.random.choice(len(im2_files), 1)[0]

        target_file = os.path.join(im1_path, im1_files[im1])
        distractor_file = os.path.join(im2_path, im2_files[im2])

        # Load selected image
        self.target = load_image(target_file)
        self.distractor = load_image(distractor_file)

        return (self.target, self.distractor)

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
        self.target_class = None

load_image("data/images/cat/0.jpg")
