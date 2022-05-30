
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.PILToTensor(),
])

train_dataset = datasets.FashionMNIST(
    "fashion_mnist",
    transform=transform,
    train=True,
    download=True
)

data = torch.tensor([x.numpy() for x, _ in train_dataset])

# reduce the problem to t-shirts
keep_target = train_dataset.class_to_idx["T-shirt/top"]
keep = train_dataset.targets == keep_target

# normalize images to [0, 1]
x = data[keep]
x = x / 255.0
x = x.reshape(x.shape[0], -1)

train_dataset = TensorDataset(x)


mean_shirt = torch.mean(x, dim=0).numpy()
mean_shirt = mean_shirt.reshape(28, 28)
plt.imshow(mean_shirt, cmap="gray")
plt.title("Mean Train Shirt")
plt.axis("off")
plt.show()


# same for the validation set
val_dataset = datasets.FashionMNIST(
    "fashion_mnist",
    transform=transform,
    train=False,
    download=False
)

data = torch.tensor([x.numpy() for x, _ in val_dataset])

keep = val_dataset.targets == keep_target

x = data[keep]
x = x / 255.0
x = x.reshape(x.shape[0], -1)

val_dataset = TensorDataset(x)
