import torch
from torch import nn
import numpy as np

class SenderLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, selected_word_prob, reward):
      return torch.mean(torch.tensor(-1 * np.log(selected_word_prob) * reward, requires_grad = True))
