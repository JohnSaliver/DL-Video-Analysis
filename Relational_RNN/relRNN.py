import torch
from torch import nn

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
        self.Q = nn.Linear(self.inputSize + self.R_Size, self.Q_Size)
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
            Qt = self.f(self.Q(torch.cat((In[:, t], Rt), 1)))
            for a, Alpha in enumerate(self.A) :
                Ha[:, t, a] = Alpha * Ha[:, t - 1, a].clone() + (1 - Alpha) * Qt
            H = torch.reshape(Ha[:, t].clone(), (batchSize, self.H_Size))
            Out[:, t] = self.f(self.O(H))
        
        return self.f_out(self.Output(self.flat(Out)))

class relational_network(nn.Module):
    def __init__(self, embedder, embedding_size):
        super().__init__()
        self.embdder = embedder
        self.embedding_size = embedding_size

        self.simi = nn.Sequential(
            [
                nn.Linear(2*embedding_size, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid(),
            ]
        )
    def forward(self, inp, support):
        emb = self.embedder(inp)
        emb = torch.cat(support, emb).reshape(inp.shape[0], 2*self.embedding_size)
        similarity = self.simi(emb)
    def trainOnSQ(self, sample, query, optim):
        # Sample : [(im, lab), ...]
        # Query : [(im, lab), ...]

        # if the problem is K-shot learning then we gotta aggregate the embedding of the K samples
        emb_support = {} #label to embedded representative
        for im, lb in sample:
            if lb in emb_support.keys():
                emb_support[lb] += [self.embdder([im])]
            else:
                emb_support[lb] = [self.embdder([im])]
        for lb in emb_support.keys(): # we average the support embeddings
            emb_support[lb] = torch.sum(emb_support[lb]), keepdim=True)/len(emb_support[lb])

        self.train()
        losses = []
        for qIm, y in query:
            similarities = torch.zeros(len(emb_support.keys()))
            targets = torch.zeros(len(emb_support.keys()))
            model.zero_grad()
            
            for ix, lb in enumerate(emb_support.keys()):
                similarities[ix] = self.forward(qIm, emb_support[lb])
                targets[ix] = torch.float(lb==y)

            loss = nn.functional.mse_loss(similarities, targets)
            loss.backward()

            optim.step()

            losses.append(loss.item())
        return np.mean(losses)