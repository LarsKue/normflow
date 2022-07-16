
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import pathlib
import numpy as np
import matplotlib.pyplot as plt

plt.rc("figure", dpi=150)


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

            # sample a random data point from each label
            validation_set = pl_module.val_data
            labels = torch.unique(validation_set.targets)

            for i, label in enumerate(labels):
                mask = (validation_set.targets == label).numpy()
                indices = np.flatnonzero(mask)
                sample = np.random.choice(indices)

                x, y = validation_set[sample]
                x = x.reshape(1, -1).cuda()
                y = y.reshape(1, -1).cuda()

                z, _ = flow.forward(x, condition=y)

                x = x.detach().cpu().numpy().reshape(28, 28)
                z = z.detach().cpu().numpy().reshape(28, 28)

                plt.subplot(5, 4, 2 * i + 1)
                plt.imshow(x, cmap="gray")
                plt.title(f"Sample")
                plt.axis("off")

                plt.subplot(5, 4, 2 * i + 2)
                plt.imshow(z, cmap="gray")
                plt.title(f"Transformed")
                plt.axis("off")

            plt.tight_layout()
            plt.show()

            # now sample a random latent point for each label
            for i, label in enumerate(labels):
                z = flow.distribution.sample((1,))
                y = F.one_hot(label, 10)

                z = z.reshape(1, -1).cuda()
                y = y.reshape(1, -1).cuda()

                x, _ = flow.inverse(z, condition=y)

                x = x.detach().cpu().numpy().reshape(28, 28)
                z = z.detach().cpu().numpy().reshape(28, 28)

                plt.subplot(5, 4, 2 * i + 1)
                plt.imshow(x, cmap="gray")
                plt.title(f"Inverse")
                plt.axis("off")

                plt.subplot(5, 4, 2 * i + 2)
                plt.imshow(z, cmap="gray")
                plt.title(f"Sample")
                plt.axis("off")

            plt.tight_layout()
            plt.show()
