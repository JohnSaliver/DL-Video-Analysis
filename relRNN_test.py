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
from torch import nn

#Subfiles imports
from RNN.RNN import RNN_classifier
from Shrec2017.ShrecDataset import ShrecDataset
from Relational_RNN.relational_network import RelationalNetwork
from Embedding.Emb_CNN import CNNEncoder

def __main__():

    dataset = ShrecDataset()
    train_data, train_target, test_data, test_target = dataset.build(one_hot=False)
    print(dataset.dataSize, dataset.seqSize, dataset.inputSize, dataset.outputSize, dataset.trainSize)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Using cuda")
    else :
        print("Using cpu")

    embedding_size = 512
    embedder = RNN_classifier(dataset.inputSize, dataset.seqSize, embedding_size, device=device)
    relNet = RelationalNetwork(embedder, embedding_size)

    lossHistory = []
    outputs = []
    target = []
    test_Score = []

    adresse = './RNN/checkpoints'

    batchSize = 100
    learningRate = 0.0001 
    epochs = 100
    optimizer = torch.optim.Adam(relNet.parameters(), lr=learningRate)

    affichage = 5
    moyennage = 10
    saving = 10

    bar = progressbar.ProgressBar(maxval=epochs)
    bar.start()
    bar.update(0)
    train_indices = np.arange(dataset.trainSize)
    for epoch in range(epochs):
        train_indices_indices = np.arange(len(train_indices))
        Qix = np.random.choice(train_indices_indices, batchSize)
        Query_indices = train_indices[Qix]
        train_indices = np.delete(train_indices, Qix)
        Six = np.random.choice(train_indices_indices, batchSize)
        Sample_indices = train_indices[Six]
        train_indices = np.delete(train_indices, Six)
        
        while len(batch)==batchSize:
            # How to do this is now the issue
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
            print(f"epoch {epoch}, reference {ref_label.item()}, loss {loss.item()}")

        
if __name__ == "__main__":
    __main__()