import torch
from torch import nn
import numpy as np

def check_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(t,r,a,f)

class Video_Analysis_Network(nn.Module):
    def __init__(self, embedder, relational, dataset=None, device='cpu'):
        super().__init__()
        self.embedder = embedder
        self.relational = relational
        self.dataset = dataset
        self.device = device
        self.embedding_size = embedder.output_shape
        if embedder.TimeDistributed:
            self.embedding_size = (dataset.seqSize,) + self.embedding_size
        self.to(device)

    def length(self, data):
        return len(self.dataset.open_data(data, video=False))

    def open_data(self, data):
        if self.dataset == None:
            return torch.from_numpy(data).to(self.device)
        else:
            return torch.from_numpy(self.dataset.open_data(data)).to(self.device)

    def forward(self, inp, support):
        emb = self.embedder(inp)
        support = support.expand((inp.shape[0], -1))
        return self.relational(emb, support)
        

    def trainSQ(self, sample, query, optim):
        # Sample : ([im, ...], [lab, ...])
        # Query : ([im, ...], [lab, ...])

        # if the problem is K-shot learning then we gotta aggregate the embedding of the K samples
        N = query[0].shape[0]
        emb_support = self._getDict(sample) # dictionary pointing from labels to sample datapoint 
        print('dic')
        similarities = torch.zeros((len(emb_support.keys()), N), device=self.device)
        targets = torch.zeros((len(emb_support.keys()), N), device=self.device)
        print(targets.shape)
        # compute predicted similarity and label for each class of the sample set
        self.train()
        self.zero_grad()
        print('embeddings')
        sample_embeddings = self._getSampleEmbeddings(emb_support,  K=sample[0].shape[0]) # compute embedding from samples and aggregate them if K>1
        print(sample_embeddings.shape)
        for ix, lb in enumerate(emb_support.keys()):
            print('la boucle')
            data = self.open_data(sample_embeddings[ix])
            lb_simi = self.forward(query[0].float(), data.reshape((1,)+data.shape).float())
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
        print('keys', emb_support.keys())
        print('le machin', (len(emb_support.keys()),) + self.embedding_size)
        sample_embeddings = torch.zeros((len(emb_support.keys()),) + self.embedding_size, device=self.device)
        print('truc', sample_embeddings.shape)
        for ix, lb in enumerate(emb_support.keys()): #compute the embeddings of the K samples
            for im in emb_support[lb]:
                with torch.no_grad():
                    print(ix)
                    print(self.length(im))
                    print(sample_embeddings[ix, :self.length(im)].shape)
                    check_memory()
                    print(self.embedder(self.open_data(im).float()).shape)
                    # sample_embeddings[lb, :self.length(im)] += self.embedder(self.open_data(im).float())/K
        return sample_embeddings
    
    def evalSQ(self, sample, query):
        # Sample : ([im, ...], [lab, ...])
        # Query : ([im, ...], [lab, ...])
        print("sample ", sample[0].shape, "query ", query[0].shape)
        # if the problem is K-shot learning then we gotta aggregate the embedding of the K samples
        N = query[0].shape[0]
        emb_support = self._getDict(sample) # dictionary pointing from labels to sample datapoint 

        similarities = torch.zeros((len(emb_support.keys()), N), device=self.device)
        targets = torch.zeros((len(emb_support.keys()), N), device=self.device)

        # compute predicted similarity and label for each class of the sample set
        self.train()
        self.zero_grad()
        sample_embeddings = self._getSampleEmbeddings(emb_support,  K=sample[0].shape[0]) # compute embedding from samples and aggregate them if K>1
        for ix, lb in enumerate(emb_support.keys()):
            data = self.open_data(sample_embeddings[ix])
            lb_simi = self.forward(query[0].float(), data.reshape((1,)+data.shape).float())
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
