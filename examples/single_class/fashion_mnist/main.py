
import pytorch_lightning as pl

import torch
import torch.nn as nn

from itertools import chain

import normflow as nf

from examples.single_class.fashion_mnist.data import train_dataset, val_dataset
from examples.single_class.fashion_mnist.model import NFModule
from examples.single_class.fashion_mnist.callbacks import PeriodicPlot
from examples.single_class.fashion_mnist.network import AutoRegressiveNetwork


def make_params_network(in_features, hidden_features, activation=nn.ReLU, num_params=2, num_layers=1):
    in_layer = nn.Linear(in_features=in_features, out_features=hidden_features)

    hidden_layers = [
        nn.Linear(in_features=hidden_features, out_features=hidden_features) for _ in range(num_layers - 1)
    ]
    activations = [activation() for _ in hidden_layers]
    dropout = [nn.Dropout() for _ in hidden_layers]

    hidden_layers = list(chain.from_iterable(zip(dropout, hidden_layers, activations)))

    out_layer = nn.Linear(in_features=hidden_features, out_features=num_params * in_features)

    return nn.Sequential(
        in_layer,
        activation(),
        *hidden_layers,
        out_layer,
    )


def make_coupling(in_features, hidden_features, activation, hidden_layers):
    split = nf.splits.EvenSplit()

    params_network = AutoRegressiveNetwork(
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        activation=activation,
    )

    coupling = nf.couplings.AffineCoupling(params_network)

    transform = nf.transforms.CouplingTransform(
        split=split,
        coupling=coupling,
    )

    return transform


def main():
    in_features = 28 * 28 // 2
    hidden_features = 512
    activation = nn.ReLU
    hidden_layers = 4

    # TODO

    transform = nf.transforms.CompositeTransform(
        nf.transforms.blocks.SCPBlock(),
        # ensure there are an odd number of reverse permutations
        # so that each input half is transformed the same number of times
        make_coupling(in_features, hidden_features, activation, hidden_layers),
        nf.permutations.ReversePermutation(),
        make_coupling(in_features, hidden_features, activation, hidden_layers),
        nf.permutations.ReversePermutation(),
        make_coupling(in_features, hidden_features, activation, hidden_layers),
        nf.permutations.ReversePermutation(),
        make_coupling(in_features, hidden_features, activation, hidden_layers),
    )

    distribution = torch.distributions.MultivariateNormal(
        loc=torch.zeros(28 * 28).cuda(),
        covariance_matrix=torch.eye(28 * 28).cuda(),
    )

    flow = nf.Flow(
        transform=transform,
        distribution=distribution,
    )

    model = NFModule(
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
