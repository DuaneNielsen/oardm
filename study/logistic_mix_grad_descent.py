import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dst
from torch.optim import Adam
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def mix_logistic_dist(pi, loc, scale):
    pi, loc, scale = torch.log_softmax(pi, -1), loc, F.softplus(scale)
    pi_dist = dst.Categorical(logits=pi)
    logistic_dist =  dst.TransformedDistribution(
        base_distribution=dst.Uniform(0., 1.),
        transforms=[dst.SigmoidTransform().inv,
                    dst.AffineTransform(loc=loc, scale=scale)]
    )
    return dst.MixtureSameFamily(pi_dist, logistic_dist)


fig, axes = plt.subplots(1, 2)
img_x, img_y = axes

ds = CIFAR10(root='~/data', transform=ToTensor())
img, label = ds[70]
img_x.imshow((img.permute(1, 2, 0)))


class ImageGen(nn.Module):
    def __init__(self, img, n_mix=10):
        super().__init__()
        self.pi = nn.Parameter(torch.rand((*img.shape, n_mix)))
        self.loc = nn.Parameter(torch.rand((*img.shape, n_mix)))
        self.scale = nn.Parameter(torch.rand((*img.shape, n_mix)))

    def forward(self):
        return mix_logistic_dist(self.pi, self.loc, self.scale)


model = ImageGen(img)
optim = Adam(params=model.parameters(), lr=1e-2)


for epoch in range(1000000):
    optim.zero_grad()
    dist = model()
    lp = dist.log_prob(img)
    loss = -lp.mean()
    loss.backward()
    optim.step()
    print(loss.item())

    if epoch % 10 == 0:
        y = dist.sample()
        img_y.clear()
        img_y.imshow(y.permute(1, 2, 0).clamp(0, 1))
        plt.pause(0.05)