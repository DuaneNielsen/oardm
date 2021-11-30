import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import one_hot
from torch.distributions import Categorical, OneHotCategorical
from matplotlib import pyplot as plt
from pl_bolts.datamodules.binary_emnist_datamodule import BinaryEMNISTDataModule
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union
from argparse import ArgumentParser, Namespace
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.argparse import from_argparse_args
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.models.vision.unet import UNet
import wandb
import numpy as np


fig, ax = plt.subplots(2, 3)
train_input_ax, train_output_ax, sample_ax = ax[0]
val_1_ax, val_2_ax, val_3_ax = ax[1]
fig.show()


def clear_ax():
    for row in ax:
        for a in row:
            a.clear()


class AODM(pl.LightningModule):
    def __init__(self, h, w, k, lr=1e-4):
        super().__init__()
        self.h, self.w, self.k = h, w, k
        self.d = self.h * self.w
        self.fc = nn.Sequential(nn.Linear(h * w * k, 1024), nn.ELU(), nn.Linear(1024, h * w * k))
        self.lr = lr
        self.tstep = 0

    def forward(self, x):
        N, H, W, K = x.shape
        x = self.fc(x.flatten(start_dim=1)).reshape(N, H, W, K)
        return torch.log_softmax(x, dim=3)

    def sample_t(self, N):
        return torch.randint(1, self.d + 1, (N, 1, 1))

    def sample_sigma(self, N):
        return torch.stack([torch.randperm(self.d).reshape(self.h, self.w) + 1 for _ in range(N)])

    def training_step(self, data):
        x, label = data[0], data[1]
        x = x.permute(0, 3, 2, 1)
        x = torch.cat((x, (1. - x)), dim=3)
        N, H, W, K = x.shape
        t = self.sample_t(N)
        sigma = self.sample_sigma(N)
        mask = sigma < t
        mask = mask.unsqueeze(-1).float()
        x_ = self(x * mask)
        C = Categorical(logits=x_)
        l = (1. - mask) * C.log_prob(torch.argmax(x, dim=3)).unsqueeze(-1)
        n = 1./(self.d - t + 1.)
        l = n * l.sum(dim=(1, 2, 3))
        return {'loss': -l.mean(), 'input': x, 'generated': x_}

    def training_step_end(self, o):
        if self.tstep % 400 == 0:
            loss, x, x_ = o['loss'], o['input'], o['generated']
            clear_ax()
            train_input_ax.imshow(x[0, :, :, 0].cpu().detach())
            train_output_ax.imshow(x_[0, :, :, 0].cpu().detach())
            sample = self.sample_one().detach()
            sample_ax.imshow(sample[:, :, 0])
            fig.canvas.draw()
        self.tstep += 1

    def validation_epoch_end(self, outputs):
        sample = self.sample_one().detach()
        val_1_ax.imshow(sample[:, :, 0])

        sample = self.sample_one().detach()
        val_2_ax.imshow(sample[:, :, 0])

        sample = self.sample_one().detach()
        val_3_ax.imshow(sample[:, :, 0])

        fig.canvas.draw()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def sample_one(self):
        x = torch.zeros(1, self.h, self.w, self.k)
        sigma = self.sample_sigma(1)
        for t in range(1, self.d+1):
            mask, current = sigma < t, sigma == t
            mask, current = mask.unsqueeze(-1).float(), current.unsqueeze(-1).float()
            x_ = OneHotCategorical(logits=self((x * mask))).sample()
            x = x * (1 - current) + x_ * current
        return x.squeeze()


if __name__ == '__main__':

    seed_everything(1234)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BinaryEMNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    dm = BinaryEMNISTDataModule(batch_size=8)
    model = AODM(28, 28, 2)
    trainer = Trainer()
    trainer.fit(model, datamodule=dm)