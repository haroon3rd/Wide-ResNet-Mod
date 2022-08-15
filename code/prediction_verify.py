# This algorithom is an implementation of Google's AutoPilot for Borge
# to work on Docker

from torchvision import datasets
import numpy as np
import os

def npy_loader(prediction_path, filename):
    sample = np.load(prediction_path, allow_pickle=True)
    print (sample.ndim)
    print (sample.shape)
    print (len(sample))
    print(type(sample))
    f = open(filename, "w")
    for i in range(len(sample)):
        max = np.argmax(sample[i], axis=0)
        if int(max) == 0:
            txt = "Airplane"
        elif int(max) == 1:
            txt = "Automobile"
        elif int(max) == 2:
            txt = "Bird"
        elif int(max) == 3:
            txt = "Cat"
        elif int(max) == 4:
            txt = "Deer"
        elif int(max) == 5:
            txt = "Dog"
        elif int(max) == 6:
            txt = "Frog"
        elif int(max) == 7:
            txt = "Horse"
        elif int(max) == 8:
            txt = "Ship"
        elif int(max) == 9:
            txt = "Truck"
        f.write(str(i) + "abcd1234.png = " + str(max) + " : " + txt + "\n")
    f.close()

dataset = datasets.DatasetFolder(
    root='./',
    loader=npy_loader,
    extensions='.npy'
)

path = os.path.abspath('./data/predictions20.npy')
npy_loader(path,"AmranLabel20.txt")