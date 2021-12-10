import torch
import torch.linalg
from torch.optim import Adam
import torch.nn as nn
from torch.nn.functional import mse_loss
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from pl_bolts.datasets import BinaryMNIST


class Radial(nn.Module):
    def __init__(self, h, w, size):
        super().__init__()
        grid = torch.stack(torch.meshgrid(torch.linspace(0, 1.0, h), torch.linspace(0, 1.0, w)))
        grid = grid.reshape(1, 2, 1, 1, size)
        noise = torch.randn_like(grid) * 1e-3
        self.cent = nn.Parameter(grid.add(noise))
        height = torch.zeros(1, 1, 1, 1, size, requires_grad=True)
        noise = torch.randn_like(height) * 1e-3
        self.bias = nn.Parameter(height.add(noise))

    def forward(self, grid):
        vec = grid.unsqueeze(-1) - self.cent
        layers = torch.linalg.vector_norm(vec, dim=1) + self.bias
        min = - torch.logsumexp(-layers * 20.0, dim=-1) / 20.0
        return min.squeeze(1)


if __name__ == '__main__':

    ds = BinaryMNIST(root='~/data', download=True)

    train_data = ds.train_data[0:50]
    train_data = distance_transform_edt(train_data == 0)
    five = torch.from_numpy(train_data[0]).float()
    H, W = five.shape
    model = Radial(H, W, H * W)
    optim = Adam(model.parameters(), lr=1e-3)
    grid = torch.stack(torch.meshgrid(torch.linspace(0, 1.0, H), torch.linspace(0, 1.0, W))).unsqueeze(0)


    plt.ion()
    fig = plt.figure()
    spec = fig.add_gridspec(2, 3)
    ax_gt = fig.add_subplot(spec[:, 0])
    ax_est = fig.add_subplot(spec[:, 1])
    ax_centers = fig.add_subplot(spec[:, 2])
    plt.pause(0.01)

    five = (five - five.mean()) / five.std()
    five_mean = five.mean()

    ax_gt.imshow(five)

    for step in range(10000):
        optim.zero_grad()
        df = model(grid)
        loss = mse_loss(df, five)
        loss.backward()
        print(f'loss {loss.item()} df mean {df.mean()} five mean {five_mean}')
        optim.step()

        if step % 10 == 0:
            ax_est.clear()
            ax_est.imshow(df.detach().cpu().squeeze(), vmin=five.min(), vmax=five.max())
            ax_centers.clear()
            ax_centers.scatter(model.cent.squeeze()[1].detach().cpu(), model.cent.squeeze()[0].detach().cpu())

            plt.pause(0.05)
