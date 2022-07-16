
import pytorch_lightning as pl

import pathlib
import numpy as np
import matplotlib.pyplot as plt

import examples.single_class.convolutional.settings as s


def plot_channels(img):
    colors = ["Red", "Green", "Blue"]
    cmaps = ["Reds_r", "Greens_r", "Blues_r"]
    plt.subplot(2, 3, 2)
    plt.imshow(img.moveaxis(0, -1))
    plt.title("Image")
    plt.axis("off")

    for i, (channel, color, cmap) in enumerate(zip(img, colors, cmaps)):
        plt.subplot(2, 3, 3 + i + 1)
        plt.imshow(channel, cmap=cmap)
        plt.title(color)
        plt.axis("off")


class PeriodicPlot(pl.Callback):
    def __init__(self, condition):
        super().__init__()
        self.condition = condition

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        if self.condition(epoch):
            path = pathlib.Path("figures")
            path.mkdir(parents=True, exist_ok=True)

            flow = pl_module.flow

            # sample a random data point
            validation_set = pl_module.val_data

            sample = np.random.randint(len(validation_set))

            x = validation_set[sample][None, ...].to(s.DEVICE)

            z, _ = flow.forward(x)

            x = x.detach().cpu().numpy().reshape(s.IMAGE_SHAPE)
            z = z.detach().cpu().numpy().reshape(s.IMAGE_SHAPE)
            x = np.moveaxis(x, 0, -1)
            z = np.moveaxis(z, 0, -1)

            plt.subplot(2, 2, 1)
            plt.imshow(x)
            plt.title(f"Sample")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(z)
            plt.title(f"Transformed")
            plt.axis("off")

            # sample a random latent point
            z = flow.distribution.sample((1,)).to(s.DEVICE)

            x, _ = flow.inverse(z)

            x = x.detach().cpu().numpy().reshape(s.IMAGE_SHAPE)
            z = z.detach().cpu().numpy().reshape(s.IMAGE_SHAPE)
            x = np.moveaxis(x, 0, -1)
            z = np.moveaxis(z, 0, -1)

            plt.subplot(2, 2, 3)
            plt.imshow(x)
            plt.title(f"Inverse")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(z)
            plt.title(f"Sample")
            plt.axis("off")

            plt.suptitle(f"Epoch {epoch}")

            plt.show()


