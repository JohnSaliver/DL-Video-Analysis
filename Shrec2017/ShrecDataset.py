import torch
from zipfile import ZipFile
import os
from os.path import isfile, join
import numpy as np
from PIL import Image
import patoolib
import cv2
print('Ca commence')
# print(os.listdir('./Shrec2017/'))
class ShrecDataset:
    def __init__(self, full=False, rescale=None):
        if full :
            self.root_datase = './Shrec2017/HandGestureDataset_SHREC2017'
            if  not('HandGestureDataset_SHREC2017' in os.listdir('./Shrec2017/')):
                print('Dataset not unziped, unziping...')
                patoolib.extract_archive("./Shrec2017/HandGestureDataset_SHREC2017.rar", outdir='./Shrec2017/', verbosity=-1)
                print("Finished unzipping...")

        else :
            self.root_datase = './Shrec2017/HandGestureDataset_SHREC2017_temp'
            if  not('HandGestureDataset_SHREC2017_temp' in os.listdir('./Shrec2017/')):
                print('Dataset not unziped, unziping...')
                with ZipFile('./Shrec2017/HandGestureDataset_SHREC2017_temp.zip', 'r') as zipObj:
                    zipObj.extractall('./Shrec2017/')
                    zipObj.close()
                print("Finished unzipping...")
        self.rescale = rescale
        self.build() # Build the Data_pointer and the Ground_truth
        self.dataSize = len(self.Data_pointer)
        self.inputSize = self.open_data(self.Data_pointer[0], video=True).shape[1:]
        self.seqSize = 171 # self.get_seqSize()
        


    def open_data(self, path, video=True, add_depth=True):
        if video :
            out = []
            exist = True
            t = 0
            while exist:
                path_image = path + '{0}_depth.png'.format(t)
                if os.path.exists(path_image):
                    I= np.array(cv2.imread(path_image, 0))
                    if self.rescale is not None:
                        I = np.array(cv2.resize(I, self.rescale, interpolation = cv2.INTER_AREA))
                    out.append(I)
                    t += 1
                else:
                    exist = False
            if add_depth:
                out = np.expand_dims(out, axis=1)
            return np.array(out)
        else:
            return np.loadtxt(path + 'skeletons_image.txt')

    def open_datas(self, paths, video=True, add_depth=True):
        ims = []
        for path in paths:
            vid = self.open_data(path)
            ims.append(np.pad(vid, [(0, 171-len(vid)), (0, 0), (0, 0), (0, 0)], mode='constant'))  
        # print(np.array(ims).shape)
        return torch.Tensor(np.array(ims))

    def build(self):
        ## Build the data pointer and the ground truth
        self.Ground_truth = []
        self.Data_pointer = []

        for idx_gesture in range(14):
            for idx_subject in range(28):
                for idx_finger in range(2):
                    for idx_essai in range(10):
                        pointer = '{0}/gesture_{1}/finger_{2}/subject_{3}/essai_{4}/'.format(self.root_datase, idx_gesture+1, idx_finger+1, idx_subject+1, idx_essai+1)
                        if os.path.exists(pointer):
                            self.Data_pointer.append(pointer)
                            self.Ground_truth.append(idx_gesture)
                        else:
                            break
        self.Data_pointer = np.array(self.Data_pointer)


    def get_seqSize(self):
        max_size = 0
        for pointer in self.Data_pointer:
            size = len(self.open_data(pointer, video=False))
            if max_size < size:
                max_size = size
        return(max_size)


    def get_data(   self, 
                    training_share=0.9, # float from 0 to 1
                    one_hot=True): 
        ## Build the target from the ground truth
        if one_hot:
            Target = np.zeros((len(self.Ground_truth), 14))
            for i in range(len(self.Ground_truth)):
                Target[i, self.Ground_truth[i]] = 1
        else:
            Target = np.array(self.Ground_truth).reshape([len(self.Ground_truth), 1])
        self.outputSize = Target.shape[1]

        ## Select train and test datasets
        self.trainSize = int(self.dataSize * training_share)

        ind_train = np.random.choice(self.dataSize, self.trainSize, replace = False)
        train_data = self.Data_pointer[ind_train]
        train_target = Target[ind_train]

        ind_test = np.delete(np.arange(self.dataSize), ind_train)
        test_data = self.Data_pointer[ind_test]
        test_target = Target[ind_test]

        
        return train_data, train_target, test_data, test_target
 

""" Test
print(os.listdir('./Shrec2017/'))
database = ShrecDataset(full=True)
train_data, train_target, test_data, test_target = database.get_data(0.9, True)
print(train_data.shape, train_target.shape, test_data.shape, test_target.shape)
"""

