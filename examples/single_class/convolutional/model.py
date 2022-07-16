
import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import normflow as nf

import examples.single_class.convolutional.settings as s


class FlowModule(pl.LightningModule):
    def __init__(self,
                 flow: nf.flows.Flow,
                 train_data: Dataset = None,
                 val_data: Dataset = None,
                 batch_size: int = s.BATCH_SIZE,
                 lr: float = s.LEARNING_RATE,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["flow", "train_data", "val_data"])
        self.flow = flow
        self.train_data = train_data
        self.val_data = val_data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.hparams.lr)
        step_size = 1
        gamma = 0.1 ** (step_size / s.MAX_EPOCHS)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=2)
        return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.to(s.DEVICE)

        z, log_prob = self.flow.forward(x)

        nll = -log_prob.mean(dim=0)

        self.log("training_nll", nll)

        return nll

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.to(s.DEVICE)

        z, log_prob = self.flow.forward(x)

        nll = -log_prob.mean(dim=0)
        self.log("validation_nll", nll)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=16,
        )

