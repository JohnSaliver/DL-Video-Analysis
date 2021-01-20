import torch
from zipfile import ZipFile
import os
from os.path import isfile, join
import numpy as np
import patoolib
from PIL import Image


class ShrecDataset:
    def __init__(self, full='False'):
        if full :
            self.root_datase = './Shrec2017/HandGestureDataset_SHREC2017'

            if  not('HandGestureDataset_SHREC2017' in os.listdir('./Shrec2017/')):
                patoolib.extract_archive("./Shrec2017/HandGestureDataset_SHREC2017.rar", outdir='./Shrec2017/', verbosity=-1)

        else :
            self.root_datase = './Shrec2017/HandGestureDataset_SHREC2017_temp'
            if  not('HandGestureDataset_SHREC2017_temp' in os.listdir('./Shrec2017/')):
                with ZipFile('./Shrec2017/HandGestureDataset_SHREC2017_temp.zip', 'r') as zipObj:
                    zipObj.extractall('./Shrec2017/')
                    zipObj.close()

        self.dataSize, self.seqSize, self.inputSize, self.outputSize, self.trainSize = (0,0,0,0,0) #Gets set in build

    def open_data(self, idx_gesture, idx_subject, idx_finger, idx_essai, video='False'):
        # Path of the gesture
        path_gesture = '{0}/gesture_{1}/finger_{2}/subject_{3}/essai_{4}/'.format(self.root_datase, idx_gesture+1, idx_finger+1, idx_subject+1, idx_essai+1)

        if os.path.isdir(path_gesture):
            if video :
                out = []
                exist = True
                t = 0
                while exist:
                    path_image = path_gesture + 'depth_{0}.png'.format(t)
                    if os.path.exists(path_image):
                        out.append(Image.open(path_image))
                        t += 1
                    else:
                        exist = False
                return True, out
            else:
                return True, np.loadtxt(path_gesture + '/skeletons_image.txt')
        else:
            return False, None

    def build(self, video=False, one_hot=True):

        Ground_truth = []
        Data = []
        for idx_gesture in range(14):
            for idx_subject in range(28):
                for idx_finger in range(2):
                    for idx_essai in range(10):
                        Exist, x = self.open_data(idx_gesture, idx_subject, idx_finger, idx_essai, video=video)
                        if not(Exist) :
                            break
                        Data.append(x)
                        Ground_truth.append(idx_gesture)

        self.dataSize = len(Data)
        for x in Data:
            if self.seqSize < len(x):
                self.seqSize = len(x)
        self.inputSize = Data[0].shape[1:]

        """ Padding to seqSize
        if video:
            Data_set = Data
        else:
            Data_set = np.zeros((self.dataSize, self.seqSize, self.inputSize))
            for i in range(self.dataSize):
                Data_set[i, :Data[i].shape[0], :] = np.reshape(Data[i], (Data[i].shape[0], Data[i].shape[1]), order = 'F')
        """

        ## Build the target from the ground truth
        if one_hot:
            Target = np.zeros((len(Ground_truth), 14))
            for i in range(len(Ground_truth)):
                Target[i, Ground_truth[i]] = 1
        else:
            Target = np.array(Ground_truth).reshape([len(Ground_truth), 1])

            
        self.outputSize = Target.shape[1]


        ## Select train and test datasets
        self.trainSize = int(self.dataSize * 0.9)

        ind_train = np.random.choice(self.dataSize, self.trainSize, replace = False)
        train_data = Data[ind_train]
        train_target = Target[ind_train]

        ind_test = np.delete(np.arange(self.dataSize), ind_train)
        test_data = Data[ind_test]
        test_target = Target[ind_test]

        return train_data, train_target, test_data, test_target

# Test
print(os.listdir('./Shrec2017/'))
database = ShrecDataset(full=True)