"""
Fonction d'embedding

Similaire au papier d'origine
On utilise un reshape pour transformer la vidéo en image
Le choix de quelle dimention inclue la temporelle est dim_choice
Si auccun choix n'est fait, la convolution s'effectue sur l'ensemble des images via TimeDistributed
"""
import torch
import torch.nn as nn

def check_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(t,r,a,f)

class Emb_CNN(nn.Module):
    """docstring for ClassName"""
    def __init__(   self,
                    input_shape,    # = (timesteps : unused if dim_concat=None, image_shape : depth, width, lenght)
                    dim_concat = None,     # int (0, 1, 2) for image_shape
                    TimeDistributed = False, # True or False
                    device="cuda"):
        
        super(Emb_CNN, self).__init__()

        self.input_reshape = list(input_shape[1:])
        print()
        self.dim_concat = dim_concat
        if dim_concat != None:
            self.input_reshape[dim_concat] = self.input_reshape[dim_concat]*input_shape[0]
        self.input_reshape = tuple(self.input_reshape)
        print('in shape', self.input_reshape)
        self.output_shape = (64, self.input_reshape[1]//16, self.input_reshape[2]//16)
        print('output_shape', self.output_shape)
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
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        
        self.to(device)

    def forward(self,x):
        length = x.size(0)
        x = x.contiguous().view((-1,) + (self.input_reshape))
        print('go')
        check_memory()
        x = self.layer1(x)
        check_memory()
        y = self.layer2(x)
        check_memory()
        y = self.layer3(y)
        check_memory()
        y = self.layer4(y)
        check_memory()
        
        if self.TimeDistributed:
            y = y.contiguous().view(length, -1, y.size(-1))
        return y
"""
from torchsummary import summary
<<<<<<< HEAD
model = Emb_CNN((-1, 3, 50, 50), dim_concat=None, TimeDistributed = True)
=======
model = Emb_CNN((100, 3, 50, 50), dim_concat=2, TimeDistributed = False)
>>>>>>> parent of 98900cb... Dataset_image added
summary(model.cuda(),(100, 3, 50, 50))"""