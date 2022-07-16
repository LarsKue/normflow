import pytorch_lightning as pl

import torch
import torch.nn as nn

import normflow as nf

from examples.single_class.fashion_mnist.data import train_dataset, val_dataset
from examples.single_class.fashion_mnist.model import ConditionalFlowModule
from examples.single_class.fashion_mnist.callbacks import PeriodicPlot
from examples.single_class.fashion_mnist.network import AutoRegressiveNetwork


def make_block(num_features, num_classes, hidden_features, hidden_layers, activation=nn.ReLU):
    split = nf.splits.EvenSplit()

    in_features = num_features // 2
    out_features = 2 * (num_features - in_features)

    params_network = AutoRegressiveNetwork(
        in_features=in_features,
        condition_features=num_classes,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        activation=activation,
    )

    coupling = nf.couplings.AffineCoupling(
        params_network=params_network,
    )

    permutation = nf.permutations.ReversePermutation()

    return nf.transforms.blocks.SCPBlock(
        split=split,
        coupling=coupling,
        permutation=permutation,
    )


def main():
    num_features = 28 * 28
    num_classes = 10
    hidden_features = 64
    hidden_layers = 2
    activation = nn.ReLU

    transform = nf.transforms.CompositeTransform(
        *[make_block(num_features, num_classes, hidden_features, hidden_layers, activation) for _ in range(4)]
    )

    distribution = torch.distributions.MultivariateNormal(
        loc=torch.zeros(num_features).cuda(),
        covariance_matrix=torch.eye(num_features).cuda(),
    )

    flow = nf.flows.Flow(
        transform=transform,
        distribution=distribution,
    )

    model = ConditionalFlowModule(
        flow=flow,
        train_data=train_dataset,
        val_data=val_dataset,
    )

    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="training_nll", save_last=True),
            PeriodicPlot(condition=lambda epoch: epoch % 1 == 0),
        ],
        gpus=-1,
        benchmark=True,
    )

    trainer.tune(model)
    print(model.hparams)

    trainer.fit(model)

    trainer.validate(ckpt_path="best")


if __name__ == "__main__":
    main()
