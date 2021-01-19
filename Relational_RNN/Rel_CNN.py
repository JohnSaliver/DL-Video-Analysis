"""
Fonction relationnelle

Similaire au papier d'origine
On suppose que l'embedding est réalisé sans TimeDistributed et donc avec dim_choice =/= None
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Rel_CNN(nn.Module):
    def __init__(self, input_shape, device="cuda"):
        super(Rel_CNN, self).__init__()

        depth = 2 * input_shape[0]
        flatten_size = 32 * (input_shape[1]//4) * (input_shape[2]//4)

        self.layer1 = nn.Sequential(
                        nn.Conv2d(depth, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.fc1 = nn.Linear(flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)

        self.to(device)

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

from torchsummary import summary
model = Rel_CNN((64, 11, 11))
summary(model.cuda(), [(64, 11, 11), (64, 11, 11)])
