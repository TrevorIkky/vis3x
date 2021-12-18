import torchvision.transforms as T
from torch.utils.data import DataLoader

import utils
from dataset import Vis3xDataset

if __name__ == "__main__":
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    dataset = Vis3xDataset(root="/storage/PCB-Compressed-L/train/",
                           augmentations=transforms)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=True)
    mean, std = utils.get_mean_and_std(data_loader)
    print(f"Mean is: {mean} \nStd is: {std}")
