import torch
import torch.nn as nn
from torch.distributions import Categorical, OneHotCategorical
from matplotlib import pyplot as plt
from pl_bolts.datamodules.binary_emnist_datamodule import BinaryEMNISTDataModule
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union
import torchvision.utils
from argparse import ArgumentParser, Namespace
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning.plugins import DDPPlugin
import wandb
import numpy as np
import metrics

plt.ion()


class Plot(pl.Callback):
    def __init__(self):
        super().__init__()
        self.r, self.c = 4, 4
        self.fig, self.ax = plt.subplots(self.r, self.c)
        self.training_ax = self.ax[0]
        self.training_images = []
        self.samples_ax = [ax for row in self.ax[1:] for ax in row]
        self.samples = []

    def draw(self):

        for img, ax in zip(self.training_images, self.training_ax):
            ax.clear()
            ax.imshow(img)

        for img, ax in zip(self.samples, self.samples_ax):
            ax.clear()
            ax.imshow(img[:, :, 0])

        self.fig.canvas.draw()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if pl_module.global_step % 1000 == 0:
            loss, x, x_ = outputs['loss'], outputs['input'], outputs['generated']
            self.training_images = [
                x[0, :, :, 0].cpu(),
                x_[0, :, :, 0].cpu(),
                x[1, :, :, 0].cpu(),
                x_[1, :, :, 0].cpu()
            ]
            self.draw()

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.samples.append(outputs['sample'])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.draw()
        del self.samples
        self.samples = []


def make_grid(image_list):
    """
    image_list: list of 28, 28, 2 images
    """
    image_list = torch.stack(image_list)
    image_list = image_list[:, :, :, 0]
    image_list = image_list.unsqueeze(1)
    image_grid = torchvision.utils.make_grid(image_list) * 255
    return image_grid


class WandbPlot(pl.Callback):
    def __init__(self):
        super().__init__()
        self.training_images = []
        self.samples = []

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if pl_module.global_step % 1000 == 0:
            loss, x, x_ = outputs['loss'], outputs['input'], outputs['generated']

            training_input = make_grid([
                x[0].cpu(),
                x[1].cpu(),
            ])

            def exp_image(image):
                return torch.exp(image + torch.finfo(image.dtype).eps)

            training_output = make_grid([
                exp_image(x_[0].cpu()),
                exp_image(x_[1].cpu())
            ])

            panel = torch.cat((training_input, training_output), dim=1)

            trainer.logger.experiment.log({
                "train_panel": wandb.Image(panel),
                })

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.samples.append(outputs['sample'])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        samples = make_grid(self.samples)
        trainer.logger.experiment.log({"samples": wandb.Image(samples)})
        del self.samples
        self.samples = []


class AODM(pl.LightningModule):
    def __init__(self, h, w, k, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.h, self.w, self.k = h, w, k
        self.d = self.h * self.w
        self.fc = nn.Sequential(nn.Linear(h * w * k, 1024), nn.ELU(), nn.Linear(1024, h * w * k))
        self.lr = lr
        self.sample_quality = metrics.BinaryMnistGenerationQuality()

    def forward(self, x):
        N, H, W, K = x.shape
        x = self.fc(x.flatten(start_dim=1)).reshape(N, H, W, K)
        return torch.log_softmax(x, dim=3)

    def sample_t(self, N):
        return torch.randint(1, self.d + 1, (N, 1, 1), device=self.device)

    def sample_sigma(self, N):
        return torch.stack([torch.randperm(self.d, device=self.device).reshape(self.h, self.w) + 1 for _ in range(N)])

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
        self.log('loss', o['loss'])

    def validation_step(self, *args, **kwargs) -> Optional:
        sample = self.sample_one()
        self.sample_quality(sample)
        return {'sample': sample.cpu()}

    def validation_epoch_end(self, outputs):
        self.log('sample_quality', self.sample_quality.compute())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def sample_one(self):
        x = torch.zeros(1, self.h, self.w, self.k, device=self.device)
        sigma = self.sample_sigma(1)
        for t in range(1, self.d + 1):
            mask, current = sigma < t, sigma == t
            mask, current = mask.unsqueeze(-1).float(), current.unsqueeze(-1).float()
            x_ = OneHotCategorical(logits=self((x * mask))).sample()
            x = x * (1 - current) + x_ * current
        return x.squeeze()

    def sample_one_seeded(self, x_seed, mask):
        x = torch.zeros(1, self.h, self.w, self.k, device=self.device)
        x[mask] = x_seed[mask]
        sigma = torch.zeros((1, self.h, self.w), dtype=torch.long, device=self.device)
        sigma[mask] = torch.arange(mask.sum()) + 1
        sigma[~mask] = torch.randperm(self.d - mask.sum(), device=self.device) + mask.sum()
        for t in range(mask.sum(), self.d + 1):
            mask, current = sigma < t, sigma == t
            mask, current = mask.unsqueeze(-1).float(), current.unsqueeze(-1).float()
            x_ = OneHotCategorical(logits=self((x * mask))).sample()
            x = x * (1 - current) + x_ * current
        return x.squeeze()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BinaryEMNISTDataModule.add_argparse_args(parser)
    parser.add_argument('--demo', default=None)
    parser.add_argument('--demo_seeded', default=None)
    parser.add_argument('--resume', default=None)
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

        wandb_logger = WandbLogger(project='oardm_binary_mnist')
        pl.seed_everything(1234)
        dm = BinaryEMNISTDataModule.from_argparse_args(args)

        model = AODM(28, 28, 2)

        trainer = pl.Trainer.from_argparse_args(args,
                                             strategy=DDPPlugin(find_unused_parameters=False),
                                             logger=wandb_logger,
                                             callbacks=[Plot(), WandbPlot()])
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
