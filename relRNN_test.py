#Package imports
import cv2
import math
import sklearn
import random
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from time import time
from scipy import misc
from scipy import ndimage
from IPython import display
from PIL import Image, ImageOps
from IPython.display import YouTubeVideo
from scipy.ndimage.filters import convolve

from sklearn.cluster import KMeans
import torch
from torch import FloatTensor, nn


#Subfiles imports
from RNN.RNN import RNN_classifier
from Shrec2017.ShrecDataset import ShrecDataset
from Relational_RNN.relational_network import RelationalNetwork
from Embedding.Emb_CNN import Emb_CNN

from Relational_RNN.Rel_CNN import Rel_CNN
from Relational_RNN.Rel_RNN import Rel_RNN
from model_CNN import Video_Analysis_Network

def _getSampleAndQuery(Indices, Classes, batchSize, K, C):
    Sample = []
    inds_bis = []
    per_classe = {}
    for classe in C:
        per_classe[classe] = 0

    for i in Indices:
        if per_classe[Classes[i, 0]]<K:
            Sample.append(i)
            per_classe[Classes[i, 0]] += 1
        else:
            inds_bis.append(i)

    if len(inds_bis) < batchSize:
        return None, None, None

    for classe in C:
        if per_classe[classe] != K:
            return None, None, None
    
    np.random.shuffle(inds_bis)
    return Sample, inds_bis[:batchSize], inds_bis[batchSize:]


def __main__():

    dataset = ShrecDataset(full=True, rescale=(30, 25))
    train_data, train_target, test_data, test_target = dataset.get_data(training_share=0.9, one_hot=False)
    print(dataset.dataSize, dataset.seqSize, dataset.inputSize, dataset.outputSize, dataset.trainSize)

    print(train_data.shape, train_target.shape, test_data.shape, test_target.shape)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Using cuda")
    else :
        print("Using cpu")

    embedding_size = 512

    #embedder = Emb_CNN((-1, 1) + dataset.inputSize, dim_concat=None, TimeDistributed = True, device=device)
    #relNet = Rel_RNN((1,) + dataset.inputSize, device=device)
    #model = Video_Analysis_Network(embedder, relNet)
    print(f"in {dataset.inputSize}")
    embedder = RNN_classifier(dataset.rescale, dataset.seqSize, embedding_size, device=device)
    relNet = RelationalNetwork(embedder, embedding_size, device=device)

    lossHistory = []
    outputs = []
    target = []
    test_Score = []

    adresse = './RNN/checkpoints'

    K = 1 #K-shot learning
    C = [2, 3, 4]
    batchSize = 16
    learningRate = 0.0005 
    epochs = 5
    optimizer = torch.optim.Adam(relNet.parameters(), lr=learningRate)

    affichage = 5
    moyennage = 10
    saving = 10

    """
    bar = progressbar.ProgressBar(maxval=epochs)
    bar.start()
    bar.update(0)
    """
    train_indices = np.where(np.isin(train_target, C), np.reshape(np.arange(dataset.trainSize), train_target.shape) , False)
    train_indices = np.array(train_indices[train_indices != [False]])
    np.random.shuffle(np.array(train_indices))

    for epoch in range(epochs):
        batch_nb = 1
        Sample_ixs, Query_ixs, train_indices_batch = _getSampleAndQuery(train_indices, Classes=train_target, batchSize=batchSize, K=K, C=C)
        while Query_ixs is not None:
            Sample_set = (dataset.open_datas(train_data[Sample_ixs]).to(device), train_target[Sample_ixs])
            Query_set = (dataset.open_datas(train_data[Query_ixs]).to(device), train_target[Query_ixs])
            batch_loss = relNet.trainSQ(sample=Sample_set, query=Query_set, optim=optimizer)
            relNet.evalSQ(sample=Sample_set, query=Query_set)
            print(f"epoch {epoch}, batch nb {batch_nb}, loss {batch_loss}")
            batch_nb+=1
            Sample_ixs, Query_ixs, train_indices_batch = _getSampleAndQuery(train_indices_batch, Classes=train_target, batchSize=batchSize, K=K, C=C)

        np.random.shuffle(np.array(train_indices))
if __name__ == "__main__":
    __main__()


""" Naive training
            batch = np.random.choice(dataset.trainSize, batchSize)
            ref_im_ix = batch[-1]
            batch = batch[:-1]
            ref_im, ref_label = train_data[ref_im_ix], train_target[ref_im_ix]
            ref_im = ref_im.reshape([1] + list(ref_im.shape))
            ref_embedding = relNet.embedder(ref_im.float())
            
            output = relNet(train_data[batch].float(), ref_embedding.float())
            y = ref_label == train_target[batch]
            loss = relNet.loss(output.float(), y.float())
            relNet.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch}, reference {ref_label.item()}, loss {loss.item()}")"""
