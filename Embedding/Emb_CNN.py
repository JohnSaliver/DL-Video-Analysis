"""
Fonction d'embedding

Similaire au papier d'origine
On utilise un reshape pour transformer la vidéo en image
Le choix de quelle dimention inclue la temporelle est dim_choice
Si auccun choix n'est fait, la convolution s'effectue sur l'ensemble des images via TimeDistributed
"""
import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y



class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(   self,
                    input_shape,    # = (timesteps, input_shape)
                    dim_concat = None,     # int (0, 1, 2) or None
                    device="cpu",
                    TimeDistributed = False):   # True or False
                    
        super(CNNEncoder, self).__init__()

        self.input_reshape = input_shape[1:len(input_shape)]
        self.dim_concat = dim_concat
        if dim_concat != None:
            self.input_reshape[dim_concat] = self.input_reshape[dim_concat]*input_shape[0]
        
        self.TimeDistributed = TimeDistributed

        self.layer1 = nn.Sequential(
                        nn.Conv2d(self.input_reshape[0], 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
                        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        x_reshape = x.contiguous().view(-1, self.input_reshape)
        y = self.layer1(x_reshape)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        
        if self.TimeDistributed:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y