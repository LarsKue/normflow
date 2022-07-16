
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import normflow as nf


class ConditionalFlowModule(pl.LightningModule):
    def __init__(self,
                 flow: nf.flows.Flow,
                 train_data: Dataset = None,
                 val_data: Dataset = None,
                 batch_size: int = 256,
                 lr: float = 1e-5,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["flow", "train_data", "val_data"])
        self.flow = flow
        self.train_data = train_data
        self.val_data = val_data

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow.parameters(), lr=self.hparams.lr)

    def forward_train(self, x, y):
        batch_size = x.shape[0]

        # flatten input
        x = x.reshape(batch_size, -1)

        z, log_prob = self.flow.forward(x, condition=y)

        return log_prob

    def inverse_train(self, x, y):
        batch_size = x.shape[0]
        # sample
        z = self.flow.distribution.sample((batch_size,))

        x, log_prob = self.flow.inverse(z, condition=y)

        return log_prob

    def training_step(self, batch, batch_idx):
        x, y = batch

        log_prob_forward = self.forward_train(x, y)
        # log_prob_inverse = self.inverse_train(x, y)

        # log_prob = torch.cat((log_prob_forward, log_prob_inverse), dim=0)
        log_prob = log_prob_forward

        # the loss is the negative log likelihood
        nll = -log_prob.mean(dim=0)
        self.log("training_nll", nll)

        return nll

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1)
        z, log_prob = self.flow.forward(x, condition=y)
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
            num_workers=16,
            pin_memory=True,
        )

