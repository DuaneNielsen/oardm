import torch
from pl_bolts.datamodules import BinaryEMNISTDataModule
from matplotlib import pyplot as plt
from train_binary_mnist import AODM
import torchmetrics


def compute_dist(dm):

    dm.setup()
    dl = dm.val_dataloader()

    mu, std, N = torch.zeros((dm.num_classes, *dm.dims)), torch.zeros((dm.num_classes, *dm.dims)), torch.zeros(dm.num_classes, 1, 1, 1)

    for x, labels in dl:
        for i in range(dm.num_classes):
            indx = labels == i
            N[i] += indx.numel()
            mu[i] += x[indx].mean(dim=0) * N[i]

    mu = mu / N
    return mu.permute(0, 1, 3, 2)


dm = BinaryEMNISTDataModule(batch_size=4028)
mu = compute_dist(dm)


class BinaryMnistGenerationQuality(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, ignore_index=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.add_state("best", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state('mu', default=mu)

    def update(self, sample: torch.Tensor):

        cls_probs = []
        #sample = sample.permute(2, 0, 1)
        eps = torch.finfo(mu.dtype).eps
        for i in range(dm.num_classes):
            prob = sample[0] * self.mu[i] + (1 - sample[0]) * (1 - self.mu[i])
            cls_probs += [prob + eps]

        cls_probs = torch.stack(cls_probs)
        cls_probs = torch.logsumexp(cls_probs, dim=(1, 2, 3))
        self.best = cls_probs.max()

    def compute(self):
        return self.best


if __name__ == '__main__':

    model = AODM.load_from_checkpoint('lightning_logs/version_48/checkpoints/epoch=4979-step=6023879.ckpt')

    mu = compute_dist(dm)

    plt.ion()

    fig = plt.figure()
    spec = fig.add_gridspec(ncols=10, nrows=4)
    ax_mu, ax_prob = [], []
    for col in range(dm.num_classes):
        ax_mu += [fig.add_subplot(spec[3, col])]
        ax_prob += [fig.add_subplot(spec[2, col])]
    ax_sample = fig.add_subplot(spec[0:2, 0:5])
    ax_bar = fig.add_subplot(spec[0:2, 5:])

    eps = torch.finfo(mu.dtype).eps

    while True:
        sample = model.sample_one().permute(2, 0, 1)
        cls_probs = []

        for i in range(dm.num_classes):
            prob = sample[0] * mu[i] + (1 - sample[0]) * (1 - mu[i])
            cls_probs += [prob + eps]
            ax_mu[i].clear()
            ax_mu[i].imshow(mu[i, 0], origin='upper')
            ax_prob[i].clear()
            ax_prob[i].imshow(prob[0, :, :])
        cls_probs = torch.stack(cls_probs)
        cls_probs = torch.logsumexp(cls_probs, dim=(1, 2, 3))

        ax_sample.clear()
        ax_sample.imshow(sample[0])
        ax_bar.clear()
        color = ['blue' for _ in range(dm.num_classes)]

        topk, top_idx = torch.topk(cls_probs, 3)
        for i in top_idx:
            color[i] = 'orange'
        i = torch.argmax(cls_probs)
        color[i] = 'red'

        ax_bar.bar(range(dm.num_classes), cls_probs, color=color)
        ax_bar.set_xticks(range(dm.num_classes))

        fig.canvas.draw()
        plt.pause(1.0)