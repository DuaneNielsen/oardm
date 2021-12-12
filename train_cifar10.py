import torch
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from typing import Any, Optional
import torchvision.utils
from argparse import ArgumentParser
from torchmetrics.image.fid import FID
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning.plugins import DDPPlugin
import wandb
from pathlib import Path
import torch.distributions as dst
import torch.nn.functional as F


class Plot(pl.Callback):
    def __init__(self):
        super().__init__()
        self.r, self.c = 4, 6
        self.fig, self.ax = plt.subplots(self.r, self.c)
        self.training_ax = self.ax[0]
        self.training_images = []
        self.samples_ax = [ax for row in self.ax[1:] for ax in row]
        self.samples = []

    def draw(self):

        for img, ax in zip(self.training_images, self.training_ax):
            ax.clear()
            ax.imshow(img.permute(1, 2, 0).clamp(0., 1.))

        for img, ax in zip(self.samples, self.samples_ax):
            ax.clear()
            ax.imshow(img.permute(1, 2, 0).clamp(0., 1.))

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
            loss, x, x_m, x_ = outputs['loss'], outputs['input'], outputs['masked_input'], outputs['generated']
            self.training_images = [
                x[0].cpu(),
                x_m[0].cpu(),
                x_[0].cpu().clamp(0., 1.),
                x[1].cpu(),
                x_m[1].cpu(),
                x_[1].cpu().clamp(0., 1.)
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
        self.samples += [s for s in outputs['sample']]

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.draw()
        del self.samples
        self.samples = []


def make_grid(image_list):
    image_grid = torchvision.utils.make_grid(torch.stack(image_list))
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
            loss, x, x_m, x_ = outputs['loss'], outputs['input'], outputs['masked_input'], outputs['generated']

            training_input = make_grid([
                x[0].cpu().clamp(0, 1),
                x[1].cpu().clamp(0, 1)
            ])

            masked_input = make_grid([
                x_m[0].cpu().clamp(0, 1),
                x_m[1].cpu().clamp(0, 1)
            ])

            training_output = make_grid([
                x_[0].cpu().clamp(0, 1),
                x_[1].cpu().clamp(0, 1)
            ])

            panel = torch.cat((training_input, masked_input, training_output), dim=1)

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
        self.samples += [s for s in outputs['sample']]

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        samples = make_grid(self.samples)
        trainer.logger.experiment.log({"samples": wandb.Image(samples)})
        del self.samples
        self.samples = []


def mix_logistic_dist(pi, loc, scale):
    pi, loc, scale = torch.log_softmax(pi, -1), loc, F.softplus(scale)
    pi_dist = dst.Categorical(logits=pi)
    logistic_dist = dst.TransformedDistribution(
        base_distribution=dst.Uniform(low=torch.zeros(loc.shape, device=loc.device),
                                      high=torch.ones(loc.shape, device=loc.device)),
        transforms=[dst.SigmoidTransform().inv,
                    dst.AffineTransform(loc=loc, scale=scale)]
    )
    # return the distribution and it's mean
    return dst.MixtureSameFamily(pi_dist, logistic_dist), torch.sum(pi.exp() * loc, dim=-1)


class AODM(pl.LightningModule):
    def __init__(self, c, h, w, num_mix, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.c, self.h, self.w, self.num_mix = c, h, w, num_mix
        self.d = self.c * self.h * self.w
        self.unet = UNet(num_classes=self.c * 3 * num_mix, input_channels=3)
        self.lr = lr
        self.fid = FID()

    def forward(self, x):
        N = x.shape[0]
        x_ = self.unet(x).reshape(N, self.c, self.h, self.w, self.num_mix * 3)
        pi, loc, scale = x_.chunk(3, dim=-1)
        return mix_logistic_dist(pi, loc, scale)

    def sample_t(self, N):
        return torch.randint(1, self.d + 1, (N, 1, 1, 1), device=self.device)

    def sample_sigma(self, N):
        return torch.stack([torch.randperm(self.d, device=self.device).reshape(self.c, self.h, self.w) + 1 for _ in range(N)])

    def sigma_with_mask(self, mask):
        # sigma = [1.. mask_pixels || randperm (remaining pixels) ]
        sigma = torch.zeros((1, self.h, self.w), dtype=torch.long, device=self.device)
        sigma[mask] = torch.arange(mask.sum()) + 1
        sigma[~mask] = torch.randperm(self.d - mask.sum(), device=self.device) + mask.sum()
        return sigma

    def training_step(self, batch, batch_idx):
        x, label = batch[0], batch[1]
        N, C, H, W = x.shape
        t = self.sample_t(N)
        sigma = self.sample_sigma(N)
        mask = sigma < t
        masked_x = x * mask
        x_dist, x_ = self(masked_x)
        l = ~mask * x_dist.log_prob(x)
        n = 1. / (self.d - t + 1.)
        l = n * l.sum(dim=(1, 2, 3), keepdims=True)
        return {'loss': -l.mean(), 'input': x.detach(), 'masked_input': masked_x.detach(), 'generated': x_.detach()}

    def training_step_end(self, o):
        self.log('loss', o['loss'])

    def validation_step(self, batch, batch_idx) -> Optional:
        with torch.no_grad():
            x, label = batch[0], batch[1]
            N, C, H, W = x.shape
            sample = self.sample(N)

            def to_uint8(img):
                return (img.clamp(0., 1.) * 255).to(dtype=torch.uint8)

            self.fid.update(to_uint8(x), real=True)
            self.fid.update(to_uint8(sample), real=False)
            return {'sample': sample.cpu()}

    def validation_epoch_end(self, outputs):
        self.log('FID', self.fid.compute())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        return [opt]

    def sample(self, N):
        x = torch.zeros(N, self.c, self.h, self.w, device=self.device)
        sigma = self.sample_sigma(N)
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
        with torch.no_grad():
            past, current = sigma < t, sigma == t
            dist, mean = self((x * past))
            x_ = dist.sample()
            x = x * ~current + x_ * current
            return x


def load_from_wandb_checkpoint(model_id_and_version):
    checkpoint_reference = f"duanenielsen/{project}/{model_id_and_version}"
    # download checkpoint locally (if not already cached)
    run = wandb.init(project=project)
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    return AODM.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", strict=False)


if __name__ == '__main__':

    project = 'oardm_cifar10'

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CIFAR10DataModule.add_argparse_args(parser)
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
            dm = CIFAR10DataModule(batch_size=1)
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
        dm = CIFAR10DataModule.from_argparse_args(args)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=100)

        if args.resume is not None:
            model = load_from_wandb_checkpoint(args.resume)
        else:
            model = AODM(c=3, h=32, w=32, num_mix=10)

        trainer = pl.Trainer.from_argparse_args(args,
                                                strategy=DDPPlugin(find_unused_parameters=False),
                                                logger=wandb_logger,
                                                callbacks=[Plot(), WandbPlot(), checkpoint_callback])

        trainer.fit(model, datamodule=dm)