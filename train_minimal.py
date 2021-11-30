import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.distributions import Categorical, OneHotCategorical
from matplotlib import pyplot as plt

# a set of six word phrases
phrases = [
    'the cat sat on the mat ',
    'the quick brown fox jumped over ',
    'four legs good two legs bad '
    ]


# map each unique word to an integer in range 0 .. V
vocab = {}
for word in "".join(phrases).split():
    vocab[word] = None

for i, key in enumerate(vocab):
    vocab[key] = i


def phrase_to_tensor(phrase):
    """

    :param phrase: 'the cat sat on the mat '  (make sure you put a space at the end)
    :return: one hot tensor, dim D, K  (D is sequence length K is vocab size)
    """
    tokens = [vocab[word] for word in phrase.split()]
    index = torch.tensor(tokens)
    return one_hot(index, len(vocab))


def tensor_to_phrase(phrase):
    """

    :param phrase: a tensor of dim 6, K
    :return: a phrase of 6 words
    """
    index = torch.argmax(phrase, dim=1)
    return [list(vocab)[i.item()] for i in index]


data = torch.stack([phrase_to_tensor(phrase) for phrase in phrases]).float()

N, D, K = data.shape

fig, ax = plt.subplots(1, 3)
train_input_ax, train_output_ax, sample_ax = ax
fig.show()


def clear_ax():
    for a in ax:
        a.clear()


class AODM(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d, self.k = d, k
        self.fc = nn.Sequential(nn.Linear(d * k, 1024), nn.ELU(), nn.Linear(1024, d * k))

    def forward(self, x):
        N, D, K = x.shape
        x = self.fc(x.flatten(start_dim=1)).reshape(N, D, K)
        return torch.log_softmax(x, dim=2)

    def l_t(self, x, mask, t):
        x = (1. - mask) * self(x * mask)
        norm_term = 1./(self.d - t + 1.)
        return norm_term * x.sum(dim=1)

    def sample_t(self, N):
        return torch.randint(1, self.d+1, (N, 1))

    def sample_sigma(self, N):
        return torch.stack([torch.randperm(self.d) + 1 for _ in range(N)])

    def train_step(self, x):
        N, D, K = x.shape
        t = self.sample_t(N)
        sigma = self.sample_sigma(N)
        mask = sigma < t
        mask = mask.unsqueeze(-1).float()
        x_ = self(x * mask)
        d = Categorical(logits=x_)
        l = (1. - mask) * d.log_prob(torch.argmax(x, dim=2)).unsqueeze(-1)
        n = 1./(self.d - t + 1.)
        l = n * l.sum(dim=(1, 2))
        return -l.mean(), x, x_

    def sample_one(self):
        x = torch.zeros(1, self.d, self.k)
        sigma = self.sample_sigma(1).squeeze()
        for t in range(1, self.d+1):
            mask, current = sigma < t, sigma == t
            mask, current = mask.unsqueeze(-1).float(), current.unsqueeze(-1).float()
            x_ = OneHotCategorical(logits=self((x * mask))).sample()
            x = x * (1 - current) + x_ * current
        return x.squeeze()


if __name__ == '__main__':

    addm = AODM(D, K)
    optim = torch.optim.Adam(addm.parameters(), lr=1e-3)

    for epoch in range(3000):
        optim.zero_grad()
        loss, x, x_ = addm.train_step(data)
        loss.backward()
        optim.step()

        clear_ax()
        train_input_ax.imshow(x[0].cpu().detach())
        train_output_ax.imshow(x_[0].cpu().detach())
        sample = addm.sample_one().detach()
        sample_ax.imshow(sample)
        fig.canvas.draw()
        print(loss.item())
        print(tensor_to_phrase(sample))