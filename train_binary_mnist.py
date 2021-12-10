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
from pathlib import Path

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
            ax.imshow(img[0, :, :])

        plt.pause(0.05)

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
                x[0, 0].cpu(),
                x_[0, 0].cpu(),
                x[1, 0].cpu(),
                x_[1, 0].cpu()
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
    image_list: list C, H, W of 2, 28, 28 images
    """
    image_list = torch.stack(image_list)
    image_list = image_list[:, 0, :, :]
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
        self.unet = UNet(num_classes=2, input_channels=2)
        self.lr = lr
        self.sample_quality = metrics.BinaryMnistGenerationQuality()

    def forward(self, x):
        x_ = self.unet(x)
        return torch.log_softmax(x_, dim=1)

    def sample_t(self, N):
        return torch.randint(1, self.d + 1, (N, 1, 1), device=self.device)

    def sample_sigma(self, N):
        return torch.stack([torch.randperm(self.d, device=self.device).reshape(self.h, self.w) + 1 for _ in range(N)])

    def sigma_with_mask(self, mask):
        # sigma = [1.. mask_pixels || randperm (remaining pixels) ]
        sigma = torch.zeros((1, self.h, self.w), dtype=torch.long, device=self.device)
        sigma[mask] = torch.arange(mask.sum()) + 1
        sigma[~mask] = torch.randperm(self.d - mask.sum(), device=self.device) + mask.sum()
        return sigma

    def training_step(self, batch, batch_idx):
        x, label = batch[0], batch[1]
        x = torch.cat((x, (1. - x)), dim=1)
        N, K, H, W = x.shape
        t = self.sample_t(N)
        sigma = self.sample_sigma(N)
        mask = sigma < t
        mask = mask.unsqueeze(1).float()
        x_ = self(x * mask)
        C = Categorical(logits=x_.permute(0, 2, 3, 1))
        l = (1. - mask) * C.log_prob(torch.argmax(x, dim=1)).unsqueeze(1)
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
        return [opt]

    def sample_one(self):
        x = torch.zeros(2, self.k, self.h, self.w, device=self.device)
        sigma = self.sample_sigma(1)
        for t in range(1, self.d + 1):
            x = self.sample_step(x, t, sigma)
        return x.squeeze()

    def sample_one_seeded(self, x_seed, mask):

        # add masked pixels to seed
        x = torch.zeros(1, self.k, self.h, self.w, device=self.device)
        x[mask] = x_seed[mask]
        sigma = self.sigma_with_mask(mask)

        for t in range(mask.sum(), self.d + 1):
            x = self.sample_step(x, t, sigma)
        return x

    def sample_step(self, x, t, sigma):
        """
        Performs one step of the noise reversal transition function in order sigma at time t
        x: the current state
        t: the current timestep
        sigma: the order
        """
        past, current = sigma < t, sigma == t
        past, current = past.unsqueeze(1).float(), current.unsqueeze(1).float()
        logprobs = self((x * past))
        x_ = OneHotCategorical(logits=logprobs.permute(0, 2, 3, 1)).sample().permute(0, 3, 1, 2)
        x = x * (1 - current) + x_ * current
        return x


def load_from_wandb_checkpoint(model_id_and_version):
    checkpoint_reference = f"duanenielsen/{project}/{model_id_and_version}"
    # download checkpoint locally (if not already cached)
    run = wandb.init(project=project)
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    return AODM.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")


if __name__ == '__main__':

    project = 'oardm_binary_mnist'

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BinaryEMNISTDataModule.add_argparse_args(parser)
    parser.add_argument('--demo', default=None)
    parser.add_argument('--demo_seeded', default=None)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    wandb_logger = WandbLogger(project=project, log_model='all')

    if args.demo is not None:

        model = load_from_wandb_checkpoint(args.demo)
        fig = plt.figure()
        spec = fig.add_gridspec(4, 4)
        sample_ax = fig.add_subplot(spec[0:3, :])
        progress_ax = fig.add_subplot(spec[3, :])

        while True:
            x = torch.zeros(2, model.k, model.h, model.w, device=model.device)
            sigma = model.sample_sigma(1)

            sample_ax.clear()
            sample_ax.imshow(x[0, 0])
            progress_ax.clear()
            plt.pause(0.01)

            for t in range(1, model.d + 1):
                x = model.sample_step(x, t, sigma)
                if t % 100 == 0:
                    sample_ax.clear()
                    sample_ax.imshow(x[0, 0], origin='lower')
                if t % 20 == 0:
                    progress_ax.clear()
                    progress_ax.barh(1, t)
                    progress_ax.set_xlim((0, model.d))
                    plt.pause(0.01)
            sample_ax.clear()
            sample_ax.imshow(x[0, 0])
            plt.pause(5.00)

    elif args.demo_seeded is not None:

        model = load_from_wandb_checkpoint(args.demo_seeded)
        fig = plt.figure()
        spec = fig.add_gridspec(4, 4)
        sample_ax = fig.add_subplot(spec[0:3, :])
        progress_ax = fig.add_subplot(spec[3, :])

        while True:
            dm = BinaryEMNISTDataModule(batch_size=1)
            dm.setup('test')
            ds = dm.test_dataloader()
            for batch in ds:
                # sample an image from batch
                x_seed, label = batch[0], batch[1]
                x_seed = torch.cat((x_seed, (1. - x_seed)), dim=1)

                # mask the bit we want to seed with
                mask = torch.zeros(model.h, model.w, dtype=torch.bool).squeeze(1)
                mask[model.h // 2:, :] = True
                mask = mask.reshape(model.h, model.w)

                # push the seed to the initial state, and
                x = torch.zeros(2, model.k, model.h, model.w, device=model.device)
                x[:, :, mask] = x_seed[:, :, mask]

                # construct a sigma that puts masked pixels at the start of sequence
                sigma = model.sigma_with_mask(mask.unsqueeze(0))

                # t starts at the non-mask pixels
                for t in range(mask.sum(), model.d + 1):
                    x = model.sample_step(x, t, sigma)

                    if t % 100 == 0:
                        sample_ax.clear()
                        sample_ax.imshow(x[0, 0])
                    if t % 20 == 0:
                        progress_ax.clear()
                        progress_ax.barh(1, t)
                        progress_ax.set_xlim((0, model.d))
                        plt.pause(0.01)
                sample_ax.clear()
                sample_ax.imshow(x[0, 0])
                plt.pause(5.00)

    else:

        pl.seed_everything(1234)
        dm = BinaryEMNISTDataModule.from_argparse_args(args)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=100)

        if args.resume is not None:
            model = load_from_wandb_checkpoint(args.resume)
        else:
            model = AODM(28, 28, 2)

        trainer = pl.Trainer.from_argparse_args(args,
                                                strategy=DDPPlugin(find_unused_parameters=False),
                                                logger=wandb_logger,
                                                callbacks=[Plot(), WandbPlot(), checkpoint_callback])

        trainer.fit(model, datamodule=dm)