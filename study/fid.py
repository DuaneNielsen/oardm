import torch
import torchmetrics
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor


if __name__ == '__main__':

    ds = CIFAR10(root='..')
    real = torch.stack([to_tensor(ds[i][0]) for i in range(10)])
    real = (real * 255).to(dtype=torch.uint8)
    fake = (torch.rand(10, 3, 32, 32) * 255).to(dtype=torch.uint8)
    fid = torchmetrics.image.FID()
    fid.update(real, real=True)
    fid.update(fake, real=False)
    print(fid.compute())

    fid.update(real, real=True)
    fid.update(real, real=False)
    print(fid.compute())
