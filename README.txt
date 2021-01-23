
Projet repository for the 3rd year final project at ENSEA:
Learning to Compare: Relation Network for Video Classification

Authors
Jeremy Gatineau and Mario Larsen


To run training, you first need to make sure the SHREC 2017 dataset is present in the Shrec2017 folder, if it is not you may download it from http://www-rech.telecom-lille.fr/shrec2017-hand/. 
You may also want to unzip it manually, python will do it programatically if it doesn't detect the unzipped folder but is rather slow at it as it only uses 1 core.

To run training for the relational network you can run the relRNN_test.py file on a python terminal, the hyperparameters are defined directly in the file.
You may also run RNN_test.py which trains a simple RNN on skeleton data from SHREC without K-shot learning.

After each batch, the algorithm will update the corresponding pickle file with the training loss, training and evaluation accuracy for that batch it can easily be loaded as a dictionary with the pickle.load() method.
