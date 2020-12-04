import torch
from zipfile import ZipFile
import os
from os.path import isfile, join
import numpy as np

print(os.listdir('./Shrec2017/'))

class ShrecDataset:
    def __init__(self):
        if  not('HandGestureDataset_SHREC2017_temp' in os.listdir('./Shrec2017/')):
            with ZipFile('./Shrec2017/HandGestureDataset_SHREC2017_temp.zip', 'r') as zipObj:
                zipObj.extractall('./Shrec2017/')
                zipObj.close()

        self.root_datase = './Shrec2017/HandGestureDataset_SHREC2017_temp'

        self.dataSize, self.seqSize, self.inputSize, self.outputSize, self.trainSize = (0,0,0,0,0) #Gets set in build

    def open_data(self, idx_gesture, idx_subject, idx_finger, idx_essai):
        # Path of the gesture
        path_gesture = '{0}/gesture_{1}/finger_{2}/subject_{3}/essai_{4}/'.format(self.root_datase, idx_gesture+1, idx_finger+1, idx_subject+1, idx_essai+1)

        if os.path.isdir(path_gesture):
            return True, np.loadtxt(path_gesture + '/skeletons_image.txt')
        else:
            return False, None

    def build(self):
        Ground_truth = []
        X = []
        for idx_gesture in range(14):
            for idx_subject in range(28):
                for idx_finger in range(2):
                    for idx_essai in range(10):
                        Exist, x = self.open_data(idx_gesture, idx_subject, idx_finger, idx_essai)
                        if not(Exist) :
                            break
                        X.append(x)
                        Ground_truth.append(idx_gesture)

        max = 0
        for x in X:
            if max < len(x):
                max = len(x)

        Data = np.zeros((len(X), max, X[0].shape[1]))
        for i in range(len(X)):
            Data[i, :X[i].shape[0], :] = np.reshape(X[i], (X[i].shape[0], X[i].shape[1]), order = 'F')

        Target = np.zeros((len(Ground_truth), 14))
        for i in range(len(Ground_truth)):
            Target[i, Ground_truth[i]] = 1

        self.dataSize, self.seqSize, self.inputSize = Data.shape
        self.outputSize = Target.shape[1]
        self.trainSize = int(self.dataSize * 0.9)

        ind_train = np.random.choice(self.dataSize, self.trainSize, replace = False)
        train_data = torch.from_numpy(Data[ind_train])
        train_target = torch.from_numpy(Target[ind_train])

        ind_test = np.delete(np.arange(self.dataSize), ind_train)
        test_data = torch.from_numpy(np.array(Data[ind_test]))
        test_target = Target[ind_test]

        return train_data, train_target, test_data, test_target

dat = ShrecDataset()