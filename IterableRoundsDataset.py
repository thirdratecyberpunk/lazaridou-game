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
            self.data_list.append(self.get_all_images(img_path))

    def get_all_images(self, path):
        return [os.path.join(path, i) for i in os.listdir(path) if i.endswith('.jpg')]

    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        for x in data:
            print(x)
            image = Image.open(x)
            image = self.transform(image)
            arr = np.array(image)
            yield arr

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        # returns an iterable object containing a chain iterator for the list of urls
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

def main():
    img_dirs = ["mice-10", "dog-10", "cat-10"]
    data_dir = "data"

    iterable_dataset = IterableRoundsDataset(img_dirs, data_dir, batch_size = 2)

    loader = DataLoader(iterable_dataset, batch_size = None)

    for batch in islice(loader, 5):
            # reshapes the image tensor into the expected shape
            target_display = batch[0].reshape(224, 224, 3)
            distractor_display = batch[1].reshape(224, 224, 3)

            figures = [target_display, distractor_display]
            plot_figures(figures, 1, 2)

if __name__ == '__main__':
    main()
