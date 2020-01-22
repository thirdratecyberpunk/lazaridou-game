import torch
from torch import nn
import numpy as np

class ReceiverLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, selected_image_prob, reward):
      return torch.mean(torch.tensor(-1 * np.log(selected_image_prob) * reward, requires_grad = True))
