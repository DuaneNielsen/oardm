import torch
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from typing import Any, Optional
from torchvision.utils import make_grid
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
from distribution_utils import discretized_mix_logistic_rgb as disc_mix_logistic_rgb
from distribution_utils import sample_from_discretized_mix_logistic_rgb as sample_disc_mix_logistic_rgb
from tqdm import tqdm
from math import log
import logging
from models.unet_plus_attn.model import get_vanila_unet_model
from models.unet_plus_attn.model import get_unet_attention_decoder
from models.unet_plus_attn.model import get_unet_attention_with_skip_connections_decoder
from models.unet_plus_attn.model import get_unet_depthwise_encoder_attention_with_skip_connections_decoder



logger = logging.getLogger('lightning')

rescale_color = lambda x: (x + 1.) / 2.


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
            x, x_m, x_ = rescale_color(x), rescale_color(x_m), rescale_color(x_)
            self.training_images = [
                x[0].cpu(),
                x_m[0].cpu(),
                x_[0].cpu().clamp(0., 1.),
                x[1].cpu(),
                x_m[1].cpu(),
                x_[1].cpu().clamp(0., 1.)
            ]
            self.draw()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.draw()


class WandbPlot(pl.Callback):
    def __init__(self):
        super().__init__()
        self.training_images = []
        self.samples = []

    def detect_and_fix_nan(self, *args):
        for i, arg in enumerate(args):
            if arg.isnan().any():
                arg[arg.isnan()] = 0.  # get rid of nans
                logger.error(f'while making panel nans detected in arg {i}')
        return args

    def make_step_panel(self, x, x_m, x_):
        N = x.shape[0]
        x, x_m, x_ = rescale_color(x), rescale_color(x_m), rescale_color(x_)
        x, x_m, x_ = self.detect_and_fix_nan(x, x_m, x_)
        training_input = make_grid(x.cpu().clamp(0, 1), nrow=N)
        masked_input = make_grid(x_m.cpu().clamp(0, 1), nrow=N)
        training_output = make_grid(x_.cpu().clamp(0, 1), nrow=N)
        panel = torch.cat((training_input, masked_input, training_output), dim=1)
        return panel

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
            panel = self.make_step_panel(x, x_m, x_)
            trainer.logger.experiment.log({"train_panel": wandb.Image(panel)})

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        loss, x, x_m, x_ = outputs['loss'], outputs['input'], outputs['masked_input'], outputs['generated']
        panel = self.make_step_panel(x, x_m, x_)
        trainer.logger.experiment.log({"val_panel": wandb.Image(panel)})


def mix_logistic_dist(pi, loc, scale):
    eps = torch.finfo(scale.dtype).eps
    pi, loc, scale = torch.log_softmax(pi + eps, -1), torch.sigmoid(loc * 100.) / 100., F.softplus(scale) + eps
    pi_dist = dst.Categorical(logits=pi)
    logistic_dist = dst.TransformedDistribution(
        base_distribution=dst.Uniform(low=torch.zeros(loc.shape, device=loc.device),
                                      high=torch.ones(loc.shape, device=loc.device)),
        transforms=[dst.SigmoidTransform().inv,
                    dst.AffineTransform(loc=loc, scale=scale)]
    )
    # return the distribution and it's mean
    return dst.MixtureSameFamily(pi_dist, logistic_dist), torch.sum(pi.exp() * loc, dim=-1)


def to_uint8(img):
    return ((img.clamp(-1., 1.) + 1) * 255 / 2).to(dtype=torch.uint8)


class GenerateAndTest(pl.Callback):
    def __init__(self, sample_n):
        super().__init__()
        self.sample_n = sample_n
        self.fid = FID()
        self.initialized = False
        self.model_device = 'cpu'

    def switch_device(self, pl_module):
        """
        FID inception network uses quite a bit of gpu memory, so switch it onto gpu only when required
        """
        if self.fid.device.type == 'cpu':
            self.model_device = pl_module.device
            pl_module = pl_module.cpu()
            self.fid = self.fid.to(self.model_device)
        else:
            self.fid = self.fid.cpu()
            pl_module = pl_module.to(self.model_device)
        return pl_module

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.initialized:
            pl_module = self.switch_device(pl_module)
            for dl in trainer.val_dataloaders:
                for b in tqdm(dl):
                    self.fid.update(to_uint8(b[0]).to(self.fid.device), real=True)
            self.initialized = True
            pl_module = self.switch_device(pl_module)

        sample = pl_module.sample(self.sample_n)
        pl_module = self.switch_device(pl_module)
        self.fid.fake_features.clear()
        self.fid.update(to_uint8(sample), real=False)
        score = self.fid.compute().item()
        pl_module = self.switch_device(pl_module)
        samples = make_grid([rescale_color(s) for s in sample])
        trainer.logger.experiment.log({'FID': score, "samples": wandb.Image(samples)})


class AODM(pl.LightningModule):
    def __init__(self, h, w, num_mix, lr=1e-4, ce_coeff=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.c, self.h, self.w, self.num_mix = 3, h, w, num_mix
        self.gridsize = 1. / (256 - 1.)
        self.d = self.h * self.w
        #self.unet = UNet(num_classes=10 * num_mix, input_channels=3, features_start=160)
        #self.unet = get_vanila_unet_model(in_dim=3, out_dim=10 * num_mix)
        self.unet = get_unet_attention_decoder(in_dim=3, out_dim=10 * num_mix)

        self.lr = lr
        self.ce_coeff = ce_coeff

    def forward(self, x):
        return self.unet(x)

    def sample_t(self, N):
        return torch.randint(1, self.d + 1, (N, 1, 1, 1), device=self.device)

    def sample_sigma(self, N):
        return torch.stack([torch.randperm(self.d, device=self.device).reshape(1, self.h, self.w) + 1 for _ in range(N)])

    def sigma_with_mask(self, mask):
        # sigma = [1.. mask_pixels || randperm (remaining pixels) ]
        sigma = torch.zeros((1, self.h, self.w), dtype=torch.long, device=self.device)
        sigma[mask] = torch.arange(mask.sum()) + 1
        sigma[~mask] = torch.randperm(self.d - mask.sum(), device=self.device) + mask.sum()
        return sigma

    def _step(self, x):
        N = x.shape[0]
        t = self.sample_t(N)
        sigma = self.sample_sigma(N)
        mask = sigma < t
        masked_x = x * mask
        dist_params = self(masked_x)
        log_prob, x_ = disc_mix_logistic_rgb(x, dist_params, self.gridsize)
        l = ~mask.squeeze() * log_prob
        n = 1. / (self.d - t + 1.)
        ln = n * l.sum(dim=(1, 2), keepdims=True)
        return ln, log_prob, mask, masked_x, x_

    def training_step(self, batch, batch_idx):
        x, label = batch[0], batch[1]
        x = x * 2 - 1.  # rescale from 0 .. 1 to -1 .. 1
        ln, log_prob, mask, masked_x, x_ = self._step(x)
        loss = ln.mean() + self.ce_coeff * log_prob.mean()
        loss = -loss / self.d / log(2.)
        return {'loss': loss, 'input': x.detach(), 'masked_input': masked_x.detach(), 'generated': x_.detach()}

    def training_step_end(self, o):
        self.log('loss', o['loss'])

    def validation_step(self, batch, batch_idx) -> Optional:
        with torch.no_grad():
            x, label = batch[0], batch[1]
            x = x * 2 - 1.  # rescale from 0 .. 1 to -1 .. 1
            ln, log_prob, mask, masked_x, x_ = self._step(x)
            loss = ln.mean() + self.ce_coeff * log_prob.mean()
            loss = -loss / self.d / log(2.)
            return {'loss': loss, 'input': x.detach(), 'masked_input': masked_x.detach(), 'generated': x_.detach()}

    def validation_step_end(self, o):
        self.log('val_loss', o['loss'])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.unet.parameters(), lr=self.lr, weight_decay=0.)
        return [opt]

    def sample(self, N):
        x = torch.zeros(N, self.c, self.h, self.w, device=self.device)
        sigma = self.sample_sigma(N)
        for t in range(1, self.d + 1):
            x = self.sample_step(x, t, sigma)
        return x.squeeze()

    def sample_one_seeded(self, x_seed, mask):

        # add masked pixels to seed
        x = torch.zeros(1, self.c, self.h, self.w, device=self.device)
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
            params = self((x * past))
            seed = torch.randint(low=0, high=torch.iinfo(torch.int32).max, size=(1, ), dtype=torch.int32)
            num_mix = torch.tensor([self.num_mix], dtype=torch.int32)
            x_ = sample_disc_mix_logistic_rgb(params)
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
    parser.add_argument('--matplotlib', action='store_true', default=False)
    args = parser.parse_args()

    wandb.finish()
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

        logger.error('INIT error log')
        logger.info('INIT info log')
        logger.debug('INIT debug log')

        callbacks = [WandbPlot(), checkpoint_callback, GenerateAndTest(sample_n=64)]

        if args.matplotlib:
            callbacks.append(Plot())

        if args.resume is not None:
            model = load_from_wandb_checkpoint(args.resume)
        else:
            model = AODM(h=32, w=32, num_mix=30)

        trainer = pl.Trainer.from_argparse_args(args,
                                                strategy=DDPPlugin(find_unused_parameters=False),
                                                logger=wandb_logger,
                                                callbacks=callbacks,
                                                gradient_clip_val=100.)

        trainer.fit(model, datamodule=dm)