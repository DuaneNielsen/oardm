import torch
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToTensor
from tqdm import tqdm

if __name__ == '__main__':

    ds = CIFAR10(root='..', transform=ToTensor(), train=False)
    real = torch.stack([ds[i][0] for i in range(10)])
    real = (real * 255).to(dtype=torch.uint8)
    fake = (torch.rand(10, 3, 32, 32) * 255).to(dtype=torch.uint8)
    fid = torchmetrics.image.FID()
    fid.update(real, real=True)
    fid.update(fake, real=False)
    print(fid.compute())

    fid = torchmetrics.image.FID().cuda()
    dl = DataLoader(ds, batch_size=32, drop_last=True)

    for image, label in tqdm(dl):
        image = image.cuda()
        image = (image * 255).to(dtype=torch.uint8)
        fid.update(image[:16], real=True)

    for image, label in tqdm(dl):
        image = (image * 255).to(dtype=torch.uint8).cuda()
        fid.update(image[16:32], real=False)
        score = fid.compute()
        fid.fake_features.clear()
        print(f'test {score}')

    fids = torch.tensor([fids])
    print(fids.mean())

