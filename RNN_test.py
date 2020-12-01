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

def __main__():

    dataset = ShrecDataset()
    train_data, train_target, test_data, test_target = dataset.build()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Using cuda")
    else :
        print("Using cpu")

    model = RNN_classifier(dataset.inputSize, dataset.seqSize, dataset.outputSize, device=device)
    model.train()

    lossHistory = []
    outputs = []
    target = []
    test_Score = []

    adresse = './RNN/checkpoints'

    batchSize = 100
    learningRate = 0.0001
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    affichage = 5
    moyennage = 10
    saving = 10
    bar = progressbar.ProgressBar(maxval=epochs)
    bar.start()
    bar.update(0)

    for epoch in range(epochs):

        batch = np.random.choice(dataset.trainSize, batchSize)
        output = model.forward(train_data[batch].float().to(device))
        loss = model.loss(output, train_target[batch].float())
        model.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.append(output.to("cpu"))
        target.append(train_target[batch].to("cpu"))
        lossHistory.append(loss.to("cpu"))

        test_output = model.forward(test_data.float().to(device)).detach().numpy()
        test_Score.append(np.mean(np.argmax(test_output.to("cpu"), axis=1) == np.argmax(test_target, axis=1))*100)

        if (len(test_Score) - 1) % saving == 0 :
            path = adresse + '/{}.pt'.format(len(test_Score) - 1)
            torch.save({'epoch': len(test_Score) - 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'test_Score': test_Score[-1]}, path)
                        
        if (epoch + 1) % affichage == 0 :
            display.clear_output(wait=True)
            plt.clf()

            fig, axs = plt.subplots(2, 1, figsize=(16, 18))
            axs[0].plot(lossHistory)
            # axs[0].plot(np.convolve(lossHistory, np.ones(moyennage)/moyennage)[moyennage - 1 : - moyennage + 1])
            # axs[0].legend(['loss', 'loss moyen'])
            axs[0].set_title('Loss')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')

            axs[1].plot(test_Score)
            axs[1].set_title('Réussite du set de test')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Score  (%)')
            plt.grid(True)
            
            display.display(plt.gcf())

        torch.cuda.empty_cache()
        bar.update(epoch + 1)

        display.clear_output(wait=True)