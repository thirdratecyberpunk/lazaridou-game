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
from itertools import repeat

class IterableRoundsDataset(IterableDataset):

    def __init__(self, img_dirs, data_dir, batch_size):
        self.img_dirs = img_dirs
        self.data_dir = data_dir
        self.data_list = []
        self.batch_size = batch_size
        self.transform = transforms.Compose(
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
    def shuffled_sublists(self):
        for i in self.data_list:
            random.shuffle(i)
        return self.data_list

    def process_data(self, category, data):
        """
        Takes in an array of image filenames and yields a generator of
        transformed tensors
        """
        image = Image.open(data)
        image = self.transform(image)
        arr = np.array(image)
        yield {"arr" : arr, "category": category}

    def get_stream(self, data_list):
        """
        Returns a chain object from an iterator that applies the transformation
        to the lists of filenames
        """
        print(data_list)
        return chain.from_iterable(map(self.process_data, repeat(data_list[0]), data_list[1]))

    def get_streams(self):
        """
        Returns a iterable zip object of the chain streams of sublists
        """
        # enumerates list to get categories for each stream
        # [(0, [1,2,3,...]), (1, [1,2,3,...])]
        categorised_sublists = list(enumerate(cycle(x) for x in self.shuffled_sublists))
        # gives a tuple containing the category and the cycle of filenames to get_stream
        return zip(*[self.get_stream(list) for list in categorised_sublists])
        # return zip(*[self.get_stream([list]) for list in self.shuffled_sublists])

    def __iter__(self):
        return self.get_streams()

def main():

    random.seed(0)

    img_dirs = ["dog-1", "cat-1"]
    data_dir = "data"

    iterable_dataset = IterableRoundsDataset(img_dirs, data_dir, batch_size = 2)

    loader = DataLoader(iterable_dataset, batch_size = None)

    for batch in islice(loader, 5):
            # reshapes the image tensor into the expected shape
            target_display = batch[0]["arr"].reshape(224, 224, 3)
            distractor_display = batch[1]["arr"].reshape(224, 224, 3)

            figures = [target_display, distractor_display]
            plot_figures(figures, 1, 2)

if __name__ == '__main__':
    main()
