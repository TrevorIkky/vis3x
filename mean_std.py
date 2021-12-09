import torchvision.transforms as T
from torch.utils.data import DataLoader

import utils
from dataset import Vis3xDataset

if __name__ == "__main__":
    dataset = Vis3xDataset(root="/home/ikky/Documents/datasets/PCB-Compressed/train",
                           augmentations=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=True)
    mean, std = utils.get_mean_and_std(data_loader)
    print(f"Mean is: {mean} \n Std is: {std}")
