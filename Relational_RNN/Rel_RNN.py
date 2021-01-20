"""
Fonction relationnelle

Similaire au papier d'origine mais avec un RNN entre les CNN TimeDistributed et les dense layers
On suppose que l'embedding est réalisé avec TimeDistributed et donc sans dim_choice
"""


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Simple_RNN(nn.Module):
    def __init__(self, inputSize, outputSize, device="cuda"):
        super().__init__()
        
        self.f = nn.Tanh().to(device) # Hyperparameter 
        self.f_out = nn.Softmax(dim = 1).to(device) # nn.Sigmoid() # Hyperparameter 
        self.device = device
        self.inputSize = inputSize
        self.R_Size = 100 # Hyperparameter
        self.Q_Size = 100 # Hyperparameter
        self.A = [0, 0.25, 0.5, 0.95] # Hyperparameter
        self.H_Size = len(self.A) * self.Q_Size
        self.outputSize = outputSize # Hyperparameter
        self.loss = nn.BCELoss()

        self.R = nn.Linear(self.H_Size, self.R_Size)
        self.Q = nn.Linear(self.inputSize + self.R_Size, self.Q_Size)
        self.O = nn.Linear(self.H_Size, self.outputSize)

        self.to(device)

    def forward(self, In):
        Out = torch.zeros((In.shape[0], In.shape[1], self.outputSize), device=self.device)
        Ha = torch.zeros((In.shape[0], In.shape[1], len(self.A), self.Q_Size), device=self.device)
        for t in range(In.shape[1]):
            H = Ha[:, t - 1].contiguous().view(-1, self.H_Size)
            Rt = self.f(self.R(H))
            Qt = self.f(self.Q(torch.cat((In[:, t], Rt), 1)))
            for a, Alpha in enumerate(self.A) :
                Ha[:, t, a] = Alpha * Ha[:, t - 1, a].clone() + (1 - Alpha) * Qt
            H = Ha[:, t].view(-1, self.H_Size)
            Out[:, t] = self.f(self.O(H))
        
        return Out


class Rel_RNN(nn.Module):
    def __init__(self, input_shape, device="cuda"):
        super(Rel_RNN, self).__init__()

        depth = 2 * input_shape[0]
        
        flatten_size = 64 * (input_shape[1]//4) * (input_shape[2]//4)

        self.layer1 = nn.Sequential(
                        nn.Conv2d(depth, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.RNN = Simple_RNN(flatten_size, 128, device)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.flat = nn.Flatten()
        self.to(device)

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], 2)
        x_reshape = x.contiguous().view((-1,) + x.shape[2:])

        y = self.layer1(x_reshape)
        y = self.layer2(y)
        
        y = y.contiguous().view(x.shape[:2] + (-1,))
        out = self.RNN(y)

        out = F.relu(self.fc1(out[:, -1, :]))
        out = torch.sigmoid(self.fc2(out))
        return out

from torchsummary import summary
model = Rel_RNN((64, 11, 11))
summary(model.cuda(), [(100, 64, 11, 11), (100, 64, 11, 11)])