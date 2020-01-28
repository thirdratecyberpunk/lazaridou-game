import torch
from torch import nn
import numpy as np
from torch.nn import Module

class LossPolicy(Module):
    def __init__(self):
        super(LossPolicy,self).__init__()

    def forward(self, selected_word_prob ,reward):
        result = np.mean(np.multiply(np.transpose(np.log(selected_word_prob)), reward))
        res_tensor = torch.tensor(result, requires_grad = True)
        return res_tensor
