import torch
import torch.nn as nn
import torch.distributions as dst
from torch.optim import Adam
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def logistic_dist(loc, scale):
    return dst.TransformedDistribution(
        base_distribution=dst.Uniform(low=torch.zeros(loc.shape), high=torch.ones(loc.shape) * 255),
        transforms=[dst.AffineTransform(loc=0, scale=1 / 255.0), dst.SigmoidTransform().inv,
                    dst.AffineTransform(loc=loc, scale=scale)]
    )


fig, axes = plt.subplots(1, 2)
img_x, img_y = axes

ds = CIFAR10(root='~/data', transform=ToTensor())
img, label = ds[70]
img = img * 255
img_x.imshow((img.permute(1, 2, 0)).to(dtype=torch.uint8))


class ImageGen(nn.Module):
    def __init__(self, img, n_mix=10):
        super().__init__()
        self.pi = nn.Parameter(torch.rand((*img.shape, n_mix)))
        self.loc = nn.Parameter(torch.rand((*img.shape, n_mix)) * 255.0)
        self.scale = nn.Parameter(torch.rand((*img.shape, n_mix)) * 40.0)

    def forward(self):
        return torch.log_softmax(self.pi, dim=-1), logistic_dist(self.loc, self.scale)


model = ImageGen(img)
optim = Adam(params=model.parameters(), lr=1e-2)


for epoch in range(1000000):
    optim.zero_grad()
    pi, dist = model()
    lp = dist.log_prob(img.unsqueeze(-1))
    loss = -(pi + lp).mean()
    loss.backward()
    optim.step()
    print(loss.item())

    if epoch % 1000 == 0:
        y = dist.sample()
        y = torch.sum(y * pi.exp(), dim=-1)
        img_y.clear()
        img_y.imshow(y.to(dtype=torch.uint8).permute(1, 2, 0))
        plt.pause(0.05)