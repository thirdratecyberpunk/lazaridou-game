"""
Dataloader which returns a game round as a sample
"""
from torchvision import transforms, utils
from PIL import Image
import random
from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import islice, chain, cycle
import os
import numpy as np
from display import plot_figures

class IterableRoundsDataset(IterableDataset):

    def __init__(self, img_dirs, data_dir, batch_size):
        self.img_dirs = img_dirs
        self.data_dir = data_dir
        self.data_list = []
        self.batch_size = batch_size
        self.transform =  transform = transforms.Compose(
            [
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.ToTensor()
            ])
        for img in img_dirs:
            img_path = os.path.join(self.data_dir, 'images', img)
            images = self.get_all_images(img_path)
            self.data_list.append(random.sample(images, len(images)))

    def get_all_images(self, path):
        return [os.path.join(path, i) for i in os.listdir(path) if i.endswith('.jpg')]

    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        """
        Takes in an array of image filenames and yields a generator of
        transformed tensors
        """
        for x in data:
            image = Image.open(x)
            image = self.transform(image)
            arr = np.array(image)
            yield arr

    def get_stream(self, data_list):
        """
        Returns a chain object from an iterator that applies the transformation
        to the lists of filenames
        """
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        # create a stream for every category of images
        streams = []
        for list in self.data_list:
            # forcing this into another list so it doesn't iterate over strings
            streams.append(self.get_stream([list]))

        streams_sample = zip(random.sample(streams, self.batch_size))
        # this returns a chain object rather than sampling from the zip
        return streams_sample

        # choose a given number of streams to sample from
        # return a zip object which samples from both streams simultaneously
        # shuffled_contents = zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])
        # return shuffled_contents
    def __iter__(self):
        return self.get_streams()

def main():

    # random.seed(1)

    img_dirs = ["dog-10", "cat-10", "mice-10"]
    data_dir = "data"

    iterable_dataset = IterableRoundsDataset(img_dirs, data_dir, batch_size = 2)

    loader = DataLoader(iterable_dataset, batch_size = None)

    for batch in islice(loader, 2):
            # reshapes the image tensor into the expected shape
            target_display = batch[0].reshape(224, 224, 3)
            distractor_display = batch[1].reshape(224, 224, 3)

            figures = [target_display, distractor_display]
            plot_figures(figures, 1, 2)

if __name__ == '__main__':
    main()
