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


class Plot:
    def __init__(self):
        self.r, self.c = 4, 4
        self.fig, self.ax = plt.subplots(self.r, self.c)
        self.training_ax = self.ax[0]
        self.training_images = []
        self.samples_ax = [ax for row in self.ax[1:] for ax in row]
        self.samples = []
        self.fig.show()

    def clear_ax(self):
        for row in self.ax:
            for a in row:
                a.clear()

    def draw(self):
        for img, ax in zip(self.training_images, self.training_ax):
            ax.imshow(img)

        for img, ax in zip(self.samples, self.samples_ax):
            ax.imshow(img)

        self.fig.canvas.draw()


plot = Plot()


class AODM(pl.LightningModule):
    def __init__(self, h, w, k, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.h, self.w, self.k = h, w, k
        self.d = self.h * self.w
        self.fc = nn.Sequential(nn.Linear(h * w * k, 1024), nn.ELU(), nn.Linear(1024, h * w * k))
        self.lr = lr

    def forward(self, x):
        N, H, W, K = x.shape
        x = self.fc(x.flatten(start_dim=1)).reshape(N, H, W, K)
        return torch.log_softmax(x, dim=3)

    def sample_t(self, N):
        return torch.randint(1, self.d + 1, (N, 1, 1))

    def sample_sigma(self, N):
        return torch.stack([torch.randperm(self.d).reshape(self.h, self.w) + 1 for _ in range(N)])

    def training_step(self, batch, batch_idx):
        x, label = batch[0], batch[1]
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
        n = 1. / (self.d - t + 1.)
        l = n * l.sum(dim=(1, 2, 3))
        return {'loss': -l.mean(), 'input': x.detach(), 'generated': x_.detach()}

    def training_step_end(self, o):
        if self.global_step % 1000 == 0:
            loss, x, x_ = o['loss'], o['input'], o['generated']
            plot.clear_ax()
            plot.training_images = [
                x[0, :, :, 0].cpu(),
                x_[0, :, :, 0].cpu(),
                x[1, :, :, 0].cpu(),
                x_[1, :, :, 0].cpu()
            ]
            plot.draw()

    def validation_step(self, *args, **kwargs) -> Optional:
        pass

    def validation_epoch_end(self, outputs):
        plot.samples = [self.sample_one()[:, :, 0] for _ in range(len(plot.samples_ax))]

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def sample_one(self):
        x = torch.zeros(1, self.h, self.w, self.k)
        sigma = self.sample_sigma(1)
        for t in range(1, self.d + 1):
            mask, current = sigma < t, sigma == t
            mask, current = mask.unsqueeze(-1).float(), current.unsqueeze(-1).float()
            x_ = OneHotCategorical(logits=self((x * mask))).sample()
            x = x * (1 - current) + x_ * current
        return x.squeeze()

    def sample_one_seeded(self, x_seed, mask):
        x = torch.zeros(1, self.h, self.w, self.k)
        x[mask] = x_seed[mask]
        # fig, ax = plt.subplots(2)
        # ax[0].imshow(x[0, :, :, 0])
        # ax[1].imshow(x_seed[0, :, :, 0])
        # plt.show()
        sigma = torch.zeros((1, self.h, self.w), dtype=torch.long)
        sigma[mask] = torch.arange(mask.sum()) + 1
        sigma[~mask] = torch.randperm(self.d - mask.sum()) + mask.sum()
        for t in range(mask.sum(), self.d + 1):
            mask, current = sigma < t, sigma == t
            mask, current = mask.unsqueeze(-1).float(), current.unsqueeze(-1).float()
            x_ = OneHotCategorical(logits=self((x * mask))).sample()
            x = x * (1 - current) + x_ * current
        return x.squeeze()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BinaryEMNISTDataModule.add_argparse_args(parser)
    parser.add_argument('--demo', default=None)
    parser.add_argument('--demo_seeded', default=None)
    args = parser.parse_args()

    if args.demo is not None:
        fig = plt.figure()
        ax = fig.subplots(1)
        fig.canvas.draw()
        plt.pause(0.05)

        model = AODM.load_from_checkpoint(args.demo, h=28, w=28, k=2)

        while True:
            x = torch.zeros(1, model.h, model.w, model.k)
            sigma = model.sample_sigma(1)
            for t in range(1, model.d + 1):
                mask, current = sigma < t, sigma == t
                mask, current = mask.unsqueeze(-1).float(), current.unsqueeze(-1).float()
                x_ = OneHotCategorical(logits=model((x * mask))).sample()
                x = x * (1 - current) + x_ * current
                if t % 10 == 0:
                    ax.clear()
                    ax.imshow(x[0, :, :, 0])
                    fig.canvas.draw()
            plt.pause(3.0)

    elif args.demo_seeded is not None:
        fig = plt.figure()
        ax_sample, ax_seed, ax_mask = fig.subplots(3)
        fig.canvas.draw()
        plt.pause(0.05)

        model = AODM.load_from_checkpoint(args.demo_seeded, h=28, w=28, k=2)

        while True:
            dm = BinaryEMNISTDataModule(batch_size=1)
            dm.setup('test')
            ds = dm.test_dataloader()
            for batch in ds:
                x_seed, label = batch[0], batch[1]
                x_seed = x_seed.permute(0, 3, 2, 1)
                x_seed = torch.cat((x_seed, (1. - x_seed)), dim=3)
                mask = torch.zeros(x_seed.shape[:-1], dtype=torch.bool)
                mask[0, model.h // 2:, :] = True
                x = model.sample_one_seeded(x_seed, mask)
                ax_sample.clear(), ax_seed.clear(), ax_mask.clear()
                ax_sample.imshow(x[:, :, 0])
                ax_seed.imshow(x_seed[0, :, :, 0])
                ax_mask.imshow(mask[0, :, :])
                plt.pause(3.0)

    else:

        seed_everything(1234)
        dm = BinaryEMNISTDataModule(batch_size=8)
        model = AODM(28, 28, 2)
        trainer = Trainer(enable_checkpointing=True,
                          default_root_dir='.')
        trainer.fit(model, datamodule=dm, ckpt_path="lightning_logs/version_38/checkpoints/epoch=324-step=1949999.ckpt")
