import pytorch_lightning as pl

import torch
import torch.nn as nn

import normflow as nf

from examples.single_class.convolutional.data import train_dataset, val_dataset
from examples.single_class.convolutional.model import FlowModule
from examples.single_class.convolutional.callbacks import PeriodicPlot
from examples.single_class.convolutional.network import ConvolutionalAutoRegressiveNetwork, ParamsNetwork


import examples.single_class.convolutional.settings as s


# def make_block(channels: int):
#     return nf.transforms.blocks.GlowBlock(
#         channels=channels,
#         affine_params_network=ParamsNetwork(channels // 2, channels - channels // 2),
#     )


def make_coupling(in_channels, out_channels):
    params_network = ParamsNetwork(in_channels, out_channels)
    return nf.couplings.AffineCoupling(
        params_network=params_network,
        dim=1,
    )


def make_block():
    split = nf.splits.SizedSplit((1, 2), dim=1)

    couplings = [
        make_coupling(1, 2),
        make_coupling(2, 1),
    ]

    block = nf.transforms.CompositeTransform(
        nf.transforms.ActNorm((1, 3, 1, 1)),
        nf.transforms.blocks.SCCBlock(
            split, *couplings
        ),
    )

    return block


def test():
    pass


def main():
    test()

    transform = nf.transforms.CompositeTransform(
        *[make_block() for _ in range(s.N_LAYERS)]
    )

    print("transform")

    distribution = nf.distributions.StandardNormal(shape=s.IMAGE_SHAPE)

    print("distribution")

    flow = nf.flows.Flow(
        transform=transform,
        distribution=distribution,
    )

    print("flow")

    model = FlowModule(
        flow=flow,
        train_data=train_dataset,
        val_data=val_dataset,
    )

    print("model")

    trainer = pl.Trainer(
        max_epochs=s.MAX_EPOCHS,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="training_nll", save_last=True),
            # pl.callbacks.StochasticWeightAveraging(device=s.DEVICE),
            PeriodicPlot(condition=lambda epoch: epoch != 0),
        ],
        gpus=-1,
        benchmark=True,
        gradient_clip_val=s.GRADIENT_CLIP,
    )

    print("trainer")

    trainer.fit(model)

    print("fit")

    trainer.validate(ckpt_path="best")


if __name__ == "__main__":
    main()
