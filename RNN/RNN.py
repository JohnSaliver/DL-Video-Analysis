import torch
from torch import dtype, nn
import numpy as np

class RNN_classifier(nn.Module):
    def __init__(self, inputSize, seqSize, outputSize, device="cpu"):
        super().__init__()
        
        self.f = nn.Tanh().to(device) # Hyperparameter 
        self.f_out = nn.Softmax(dim = 1).to(device) # nn.Sigmoid() # Hyperparameter 
        self.device = device
        self.inputSize = inputSize
        self.R_Size = 100 # Hyperparameter
        self.Q_Size = 100 # Hyperparameter
        self.A = [0, 0.25, 0.5, 0.95] # Hyperparameter
        self.H_Size = len(self.A) * self.Q_Size
        self.O_Size = 100 # Hyperparameter
        self.seqSize = seqSize
        self.outputSize = outputSize
        self.loss = nn.BCELoss()

        self.R = nn.Linear(self.H_Size, self.R_Size)
        self.Q = nn.Linear(np.prod(self.inputSize) + self.R_Size, self.Q_Size)
        self.O = nn.Linear(self.H_Size, self.O_Size)
        self.flat = nn.Flatten()
        self.Output = nn.Linear(self.O_Size * self.seqSize, self.outputSize)

        self.to(device)

    def forward(self, In):
        batchSize = In.shape[0]
        Out = torch.zeros((batchSize, self.seqSize, self.O_Size), device=self.device)
        Ha = torch.zeros((batchSize, self.seqSize, len(self.A), self.Q_Size), device=self.device)

        for t in range(self.seqSize):
            H = torch.reshape(Ha[:, t - 1].clone(), (batchSize, self.H_Size))
            Rt = self.f(self.R(H))
            print("rnn shapes ", In.shape, " Rt ", Rt.shape)
            Qt = self.f(self.Q(torch.cat((In[:, t], Rt), 1)))
            for a, Alpha in enumerate(self.A) :
                Ha[:, t, a] = Alpha * Ha[:, t - 1, a].clone() + (1 - Alpha) * Qt
            H = torch.reshape(Ha[:, t].clone(), (batchSize, self.H_Size))
            Out[:, t] = self.f(self.O(H))
        
        return self.f_out(self.Output(self.flat(Out)))

