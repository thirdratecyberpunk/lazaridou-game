import torch
from torch import nn
import numpy as np
from torch.nn import Module

class LossPolicy(Module):
    def __init__(self):
        super(LossPolicy,self).__init__()

    def forward(self, selection_prob ,reward):
        # payoff is the expected value
        result = np.mean(np.multiply(np.transpose(np.log(selection_prob)), reward))
        res_tensor = torch.tensor(result, requires_grad = True)
        return res_tensor
