import torch
import torch.nn as nn
from torch import Tensor

class TanhExp(nn.Module):
    def __init__(self):
        super(TanhExp, self).__init__()

    def forward(self, input: Tensor):
        return input * torch.tanh(torch.exp(input))

def tanhexp(input):
    return input * torch.tanh(torch.exp(input))