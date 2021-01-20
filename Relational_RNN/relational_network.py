import torch
from torch import nn
import numpy as np

class RelationalNetwork(nn.Module):
    def __init__(self, embedder, embedding_size, device='cpu'):
        super().__init__()
        self.embedder = embedder
        self.embedding_size = embedding_size

        self.simi = nn.Sequential(
                nn.Linear(2*embedding_size, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid()
        )
        self.loss = nn.BCELoss()
        self.to(device)
    def forward(self, inp, support):
        emb = self.embedder(inp)
        support = support.expand((inp.shape[0], -1))
        emb = torch.cat([support, emb], 1)
        emb = emb.reshape(inp.shape[0], 2*self.embedding_size)
        return self.simi(emb)
        
    def trainSQ(self, sample, query, optim):
        # Sample : ([im, ...], [lab, ...])
        # Query : ([im, ...], [lab, ...])

        # if the problem is K-shot learning then we gotta aggregate the embedding of the K samples
        emb_support = {}
        for im, lb in zip(sample[0], sample[1]):
            lb = lb.item()
            if lb in emb_support.keys():
                emb_support[lb] += [im]
            else:
                emb_support[lb] = [im]

        sample_embeddings = torch.zeros((len(emb_support.keys()), sample[0].shape[0], self.embedding_size))
        for lb in emb_support.keys(): #compute the embeddings of the K samples
            for ix, im in enumerate(emb_support[lb]):
                sample_embeddings[ix] = self.embedder(im.reshape([1]+list(im.shape)).float())
        sample_embeddings = torch.sum(sample_embeddings, dim=1)/sample[0].shape[0]
        self.train()
        similarities = torch.zeros((len(emb_support.keys()), query[0].shape[0]))
        targets = torch.zeros((len(emb_support.keys()), query[0].shape[0]))
        self.zero_grad()
        for ix, lb in enumerate(emb_support.keys()):
            lb_simi = self.forward(query[0].float(), sample_embeddings[ix].reshape([1]+list(sample_embeddings[ix].shape)).float())
            similarities[ix] = lb_simi.reshape([query[0].shape[0]])
            targets[ix] = (lb==query[1]).float().reshape([query[0].shape[0]])

        loss = nn.functional.mse_loss(similarities, targets)
        loss.backward()

        optim.step()
        
        return loss.item()

    def testST(self, support, test):
        # Support : [(im, lab), ...]
        # Test : [im, ...]

        # Same as in train, we build the support set embeddings
        emb_support = {} #label to embedded representative
        for im, lb in support:
            if lb in emb_support.keys():
                emb_support[lb] += [self.embdder([im])]
            else:
                emb_support[lb] = [self.embdder([im])]
        for lb in emb_support.keys(): # we average the support embeddings
            emb_support[lb] = torch.sum(emb_support[lb], dim=0)/len(emb_support[lb])

        self.eval()
        test_predictions = []
        for Im in test:
            similarities = []
            for lb in enumerate(emb_support.keys()):
                similarities.append(self.forward(Im, emb_support[lb]))
            pred = emb_support.keys()[np.argmax(similarities)]
            test_predictions.append(pred)
        return test_predictions # one predicted label for every image
