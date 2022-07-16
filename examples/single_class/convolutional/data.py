
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

import pathlib
from PIL import Image

import examples.single_class.convolutional.settings as s


class GanyuFacesDataset(Dataset):
    def __init__(self, path: pathlib.Path, pattern: str):
        super().__init__()

        self.path = path
        self.pattern = pattern

        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((s.IMAGE_HEIGHT, s.IMAGE_WIDTH)),
        ])

    def __len__(self):
        return len(list(self.path.glob(self.pattern.format("*"))))

    def __getitem__(self, item):
        with Image.open(self.path / self.pattern.format(item)) as img:
            image_tensor = self.transform(img)

            return image_tensor


path = pathlib.Path("ganyu-faces")
pattern = "{}.jpg"

dataset = GanyuFacesDataset(path, pattern=pattern)

train_dataset, val_dataset = random_split(dataset, (s.N_TRAIN, s.N_VAL))

import numpy as np
import matplotlib.pyplot as plt

img = train_dataset[0].numpy()
img = np.moveaxis(img, 0, -1)

plt.imshow(img)
plt.title("Sample")
plt.axis("off")
plt.show()

with torch.no_grad():
    images = torch.stack([train_dataset[i] for i in range(len(train_dataset))], dim=0)
    mean = torch.mean(images, dim=0).numpy()

img = np.moveaxis(mean, 0, -1)
plt.imshow(img)
plt.title("Mean")
plt.axis("off")
plt.show()
