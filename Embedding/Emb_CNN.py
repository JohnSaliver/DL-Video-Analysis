"""
Fonction d'embedding

Similaire au papier d'origine
On utilise un reshape pour transformer la vidéo en image
Le choix de quelle dimention inclue la temporelle est dim_choice
Si auccun choix n'est fait, la convolution s'effectue sur l'ensemble des images via TimeDistributed
"""
import torch
import torch.nn as nn


class Emb_CNN(nn.Module):
    """docstring for ClassName"""
    def __init__(   self,
                    input_shape,    # = (timesteps, input_shape)
                    dim_concat = None,     # int (0, 1, 2) or None
                    TimeDistributed = False, # True or False
                    device="cuda"):  
                    
        super(Emb_CNN, self).__init__()

        self.input_reshape = list(input_shape[1:len(input_shape)])
        self.dim_concat = dim_concat
        if dim_concat != None:
            self.input_reshape[dim_concat] = self.input_reshape[dim_concat]*input_shape[0]
        self.input_reshape=tuple(self.input_reshape)
        
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
        
        self.to(device)

    def forward(self,x):
        x_reshape = x.contiguous().view((-1,) + (self.input_reshape))
        y = self.layer1(x_reshape)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        
        if self.TimeDistributed:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y
"""
from torchsummary import summary
model = Emb_CNN((100, 3, 50, 50), dim_concat=2, TimeDistributed = False)
summary(model.cuda(),(100, 3, 50, 50))"""