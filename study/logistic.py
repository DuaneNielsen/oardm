import torch
import torch.distributions as dst
from matplotlib import pyplot as plt


def logistic_dist(loc, scale):
    return dst.TransformedDistribution(
        base_distribution=dst.Uniform(low=torch.zeros(loc.shape), high=torch.ones(loc.shape) * 255),
        transforms=[dst.AffineTransform(loc=0, scale=1 / 255.0), dst.SigmoidTransform().inv,
                    dst.AffineTransform(loc=loc, scale=scale)]
    )


fig, axes = plt.subplots(1, 2)
pdf_ax, img_ax = axes

support = torch.arange(255)
logistic = logistic_dist(loc=torch.tensor([144.0]), scale=torch.tensor([20.0]))
probs = logistic.log_prob(support).exp()
pdf_ax.plot(support, probs, label='loc=144 scale=20')
pdf_ax.set_xlabel('sub-pixel intensity')
pdf_ax.set_ylabel('prob')
pdf_ax.set_title('logistic PDF')
pdf_ax.legend()

logistic = logistic_dist(torch.ones(3, 28, 28) * 144, torch.ones(3, 28, 28) * 40)
s = logistic.sample().to(dtype=torch.uint8)
img_ax.imshow(s.permute(1, 2, 0))
img_ax.set_title('sample image')
plt.legend()
plt.show()