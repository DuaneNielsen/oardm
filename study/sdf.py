import torch
from pl_bolts.datasets import BinaryMNIST
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

"""
Study in distance field approximation...

given a field f(x, y) = d

approximate an discrete 28 x 28 sdf by..
    
    sampling some s_x, s_y co-ordinates from a meshgrid
    
    component_dfs = []
    for each s_x, s_y:
        lookup f(s_x, s_y) = d in the gt
        compute a 28 x 28 distance field using
        g = g(x, y, s_x, s_y, d) = sqrt(( x - s_x ) ** 2 + (y - s_y) ** 2) + d
        component_df += [g]
        
    finally, combine the samples using a soft minimum
    approx_df = logsumexp(component_dfs)

"""

ds = BinaryMNIST(root='~/data', download=True)

train_data = ds.train_data[0:50]
train_data = distance_transform_edt(train_data == 0)
five = torch.from_numpy(train_data[0])
H, W = five.shape

mesh = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)))
seq = torch.randperm(five.numel()).reshape(five.shape)


def dist_f(x, y, x_o, y_o, d):
    return torch.sqrt((x - x_o) ** 2 + (y - y_o) ** 2) + d


fig = plt.figure()
spec = fig.add_gridspec(2, 3)
ax_gt = fig.add_subplot(spec[0, 0])
ax_mask = fig.add_subplot(spec[0, 1])
ax_sample = fig.add_subplot(spec[0, 2])
ax_mse = fig.add_subplot(spec[1, :])

mse = []

for t in range(five.numel()):
    mask = seq <= t
    xy = mesh[:, mask]
    d = five[xy[0], xy[1]]

    samples = []
    for i in range(t+1):
        samples += [dist_f(mesh[0], mesh[1], xy[0, i], xy[1, i], d[i])]
    samples = torch.stack(samples)

    samples = -torch.logsumexp(-samples * 20.0, dim=0)/20.0
    mse += [((five - samples) ** 2).mean().item()]

    ax_gt.clear()
    ax_gt.imshow(five)
    ax_gt.set_title('ground truth')
    ax_mask.clear()
    ax_mask.imshow(mask)
    ax_mask.set_title('mask')
    ax_sample.clear()
    ax_sample.imshow(samples, vmin=0.0, vmax=five.max())
    ax_sample.set_title('sample')
    ax_mse.clear()
    ax_mse.plot(mse)
    ax_mse.set_title('mean squared error')
    ax_mse.set_xlabel('t')
    plt.pause(0.05)

plt.show()
