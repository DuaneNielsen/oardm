# [Autoregressive Diffusion Models](https://arxiv.org/abs/2110.02037)

Minimal pytorch example

```shell
python train_minimal.py
```

train CIFAR 10 for 3000 epochs
```shell
python train_cifar10.py --gpus 2 --max_epochs 3000 --batch_size 32 --limit_val_batches 2 --num_workers 32 --check_val_every_n_epoch 50
```