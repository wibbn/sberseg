import torch

def unnormalize(tensor: torch.Tensor, mean: float, std: float):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor