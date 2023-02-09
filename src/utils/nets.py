import torch
from torch import nn
import torch.nn.functional as F


class LinearFashionMNIST(nn.Module):
  def __init__(self, num_classes : int):
    super(LinearFashionMNIST, self).__init__()

    self.flatten = nn.Flatten()
  
    self.linear_stack = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.Linear(128, num_classes)
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_stack(x)
    return logits

  
class LinearFashionMNIST_alt(nn.Module):
  def __init__(self, input_size, num_classes: int):
    super(LinearFashionMNIST_alt, self).__init__()

    self.flatten = nn.Flatten()
    self.input_layer = nn.Linear(input_size, 128)
    self.output_layer = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.flatten(x)
    #x = torch.flatten(x)
    return self.output_layer(self.input_layer(x))
