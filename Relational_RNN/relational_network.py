import torch
from torch import nn
import numpy as np

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
            emb_support[lb] = torch.sum(emb_support[lb], keepdim=True)/len(emb_support[lb])

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