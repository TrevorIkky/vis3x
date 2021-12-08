import os

from PIL import Image
from torch.utils.data import Dataset


class Vis3xDataset(Dataset):
    def __init__(self, root, augmentations):
        self.root = root
        self.augmentations = augmentations
        self.images = os.listdir(root)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.images[index])
        image = Image.open(image_path).convert("RGB")

        if self.augmentations is not None:
            image = self.augmentations(image)

        return image

    def __len__(self):
        return len(self.images)
