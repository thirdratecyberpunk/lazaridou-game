import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import os

class TargetsDataset(Dataset):
    """
    Class representing the possible targets/discriminators.
    """
    def __init__(self, data_dir, transform):
        self.vocabulary = os.listdir(data_dir)
        # list of all filenames for possible targets
        self.foldernames = [os.path.join(data_dir, f) for f in self.vocabulary]
        self.objects = {}

        # for every word in the vocabulary
        for x in self.vocabulary:
            # find all images in THAT IMAGE FOLDER
            # create a new array containing all the filenames for that word
            x_dir = data_dir + "/" + x
            filenames = []
            for path, subdirs, files in os.walk(x_dir):
                for name in files:
                    filenames.append(os.path.join(x_dir,name))
            filenames.sort()
            self.objects[x] = filenames

        # self.filenames = []
        # self.labels = []
        # # for every word in the vocabulary
        # for x in self.vocabulary:
        #     # find all images in THAT IMAGE FOLDER
        #     # create a new array containing all the filenames for that word
        #     x_dir = data_dir + "/" + x
        #     x_filenames = []
        #     for path, subdirs, files in os.walk(x_dir):
        #         for name in files:
        #             x_filenames.append(os.path.join(x_dir,name))
        #             self.labels.append(x)
        #     x_filenames.sort()
        #     self.filenames.append(x_filenames)
        # self.transform = transform

def __len__(self):
    #return size of dataset
    return len(self.filenames)

# TODO: check how PyTorch should handle this
def __getitem__(self,idx):
    return get_targets_distractors()

def get_targets_distractors(targets=1, distractors=1):
    """
    Returns a pair containing a single image (the target) and another single
    image (the distractor)
    """
    # picks two random words from the vocabulary
    # chooses one of these as the target
    # target = random.sample(self.vocabulary,targets)
    possible_vocab = self.vocabulary
    target_category = possible_vocab.pop()
    # chooses the other as the distractor
    random.shuffle(possible_vocab)
    # distractor_category = random.sample(set(self.vocabulary]) - set([target]), distractors)
    distractor_category = random.choice(set([self.vocabulary]) - set([target_category]))
    # for both of these, gets a random image
    target_image = random.choice(objects[target])
    distractor_image = random.choice(objects[distractor_category])
    target_sample = {"filename": target_image,
    "img_tensor": transform(Image.open(target_image)),
    "category": target_category}
    distractor_sample = {"filename": distractor_image,
    "img_tensor": transform(Image.open(distractor_image)),
    "category": distractor_category}
    # return the pair of images
    return {"target": target_sample, "distractor": distractor_sample}

def get_vocabulary():
    """
    Returns an array of vocabulary for this set
    """
    return self.vocabulary
