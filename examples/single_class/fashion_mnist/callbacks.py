
import pytorch_lightning as pl
import torch
import pathlib
import matplotlib.pyplot as plt


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
            # torch.manual_seed(0)
            # np.random.seed(0)
            x = pl_module.val_data.tensors[0].cuda()
            idx = torch.randint(low=0, high=len(x), size=(1,)).cuda()
            x = x[idx]
            assert x.shape == (1, 28 * 28)
            z, _ = flow.forward(x)
            assert z.shape == (1, 28 * 28)

            x = x.detach().cpu().numpy().reshape(28, 28)
            z = z.detach().cpu().numpy().reshape(28, 28)

            fig, axes = plt.subplots(2, 2, dpi=200)
            fig.suptitle(f"Epoch {epoch}")

            axes[0][0].imshow(x, cmap="gray", vmin=0, vmax=1)
            axes[0][0].set_title("Data Sample")
            axes[0][0].set_axis_off()
            axes[0][1].imshow(z, cmap="gray")
            axes[0][1].set_title("Transformed")
            axes[0][1].set_axis_off()

            # sample a random latent point
            z = flow.distribution.sample((1,))
            assert z.shape == (1, 28 * 28)
            x, _ = flow.inverse(z)
            assert x.shape == (1, 28 * 28)

            x = x.detach().cpu().numpy().reshape(28, 28)
            z = z.detach().cpu().numpy().reshape(28, 28)

            # plot both the latent sample and the inverse
            axes[1][0].imshow(x, cmap="gray")
            axes[1][0].set_title("Inverse")
            axes[1][0].set_axis_off()
            axes[1][1].imshow(z, cmap="gray")
            axes[1][1].set_title("Latent Sample")
            axes[1][1].set_axis_off()

            plt.tight_layout()

            plt.savefig(path / f"{epoch}.png")

            # disable this if your plt.show is blocking
            plt.show()

            # invert the mean
            z = flow.distribution.mean.unsqueeze(0)
            assert z.shape == (1, 28 * 28)
            x, _ = flow.inverse(z)
            assert x.shape == (1, 28 * 28)

            x = x.detach().cpu().numpy().reshape(28, 28)

            plt.imshow(x, cmap="gray", vmin=0, vmax=1)
            plt.title("Learned Mean")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(path / f"mean_{epoch}.png")
            plt.show()
