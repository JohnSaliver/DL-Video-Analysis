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
        self.device = device
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
        N = query[0].shape[0]
        emb_support = self._getDict(sample) # dictionary pointing from labels to sample datapoint 
        sample_embeddings = self._getSampleEmbeddings(emb_support,  K=sample[0].shape[0]) # compute embedding from samples and aggregate them if K>1
        
        similarities = torch.zeros((len(emb_support.keys()), N), device=self.device)
        targets = torch.zeros((len(emb_support.keys()), N), device=self.device)

        # compute predicted similarity and label for each class of the sample set
        self.train()
        self.zero_grad()
        for ix, lb in enumerate(emb_support.keys()):
            lb_simi = self.forward(query[0].float(), sample_embeddings[ix].reshape([1]+list(sample_embeddings[ix].shape)).float())
            similarities[ix] = lb_simi.reshape([N])
            targets[ix] = (lb==query[1]).float().reshape([N])
        # compute the loss and perform optimization step
        loss = nn.functional.mse_loss(similarities, targets)
        loss.backward()
        optim.step()
        return loss.item()
    
    def _getDict(self, sample):
        # Sample : ([im, ...], [lab, ...])
        emb_support = {}
        for im, lb in zip(sample[0], sample[1]):
            lb = lb.item()
            if lb in emb_support.keys():
                emb_support[lb] += [im]
            else:
                emb_support[lb] = [im]
        return emb_support

    def _getSampleEmbeddings(self, emb_support, K):
        sample_embeddings = torch.zeros((len(emb_support.keys()), K, self.embedding_size), device=self.device)
        for lb in emb_support.keys(): #compute the embeddings of the K samples
            for ix, im in enumerate(emb_support[lb]):
                sample_embeddings[ix] = self.embedder(im.reshape([1]+list(im.shape)).float())
        sample_embeddings = torch.sum(sample_embeddings, dim=1)/K
        return sample_embeddings
    
    def evalSQ(self, sample, query):
        # Sample : ([im, ...], [lab, ...])
        # Query : ([im, ...], [lab, ...])
        print("sample ", sample[0].shape, "query ", query[0].shape)
        # if the problem is K-shot learning then we gotta aggregate the embedding of the K samples
        N = query[0].shape[0]
        emb_support = self._getDict(sample) # dictionary pointing from labels to sample datapoint 
        sample_embeddings = self._getSampleEmbeddings(emb_support,  K=sample[0].shape[0]) # compute embedding from samples and aggregate them if K>1
        
        similarities = torch.zeros((len(emb_support.keys()), N), device=self.device)
        targets = torch.zeros((len(emb_support.keys()), N), device=self.device)

        # compute predicted similarity and label for each class of the sample set
        self.train()
        self.zero_grad()
        for ix, lb in enumerate(emb_support.keys()):
            lb_simi = self.forward(query[0].float(), sample_embeddings[ix].reshape([1]+list(sample_embeddings[ix].shape)).float())
            similarities[ix] = lb_simi.reshape([N])
            targets[ix] = (lb==query[1]).float().reshape([N])
        # compute the loss and perform optimization step
        loss = nn.functional.mse_loss(similarities, targets)
        print(f"similarities {similarities}")
        predictions = np.argmax(similarities, axis=0)
        print(f"predictions {predictions}")
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
