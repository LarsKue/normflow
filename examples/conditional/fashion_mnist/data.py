
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])

target_transform = transforms.Compose([
    lambda y: torch.LongTensor([y]),
    lambda y: F.one_hot(y, 10),
    lambda y: torch.squeeze(y),
])

train_dataset = datasets.FashionMNIST(
    "fashion_mnist",
    transform=transform,
    target_transform=target_transform,
    train=True,
    download=True
)

# find mean for each target label
unique_targets = torch.unique(train_dataset.targets)
unique_means = []

for target in unique_targets:
    data = train_dataset.data[train_dataset.targets == target].to(torch.float64)

    mean = torch.mean(data, dim=0)

    unique_means.append(mean)

# plot the unique means
nrows = 2
ncols = 5

fig, axes = plt.subplots(nrows, ncols, dpi=200)
axes = np.array(axes)

for ax, target, mean in zip(axes.flat, unique_targets, unique_means):
    ax.imshow(mean, cmap="gray")
    ax.set_axis_off()

    target_name = train_dataset.classes[target.item()]
    ax.set_title(target_name)

fig.suptitle("Means For Each Target Label")
plt.tight_layout()
plt.show()


# # same for the validation set
val_dataset = datasets.FashionMNIST(
    "fashion_mnist",
    transform=transform,
    target_transform=target_transform,
    train=False,
    download=False
)
