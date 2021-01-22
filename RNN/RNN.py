import torch
from torch import dtype, max_pool2d, nn
import numpy as np
from torch.nn.modules.linear import Linear

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
        self.emb_size = 750
        
        self.image_embedding = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.R = nn.Linear(self.H_Size, self.R_Size)
        self.Q = nn.Linear(self.emb_size + self.R_Size, self.Q_Size)
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
            im_embed = self.image_embedding(In[:, t])
            Qt = self.f(self.Q(torch.cat((im_embed, Rt), 1)))
            for a, Alpha in enumerate(self.A) :
                Ha[:, t, a] = Alpha * Ha[:, t - 1, a].clone() + (1 - Alpha) * Qt
            H = torch.reshape(Ha[:, t].clone(), (batchSize, self.H_Size))
            Out[:, t] = self.f(self.O(H))
        
        return self.f_out(self.Output(self.flat(Out)))

