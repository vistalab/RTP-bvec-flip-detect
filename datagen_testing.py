import numpy as np
from preprocess import dataLoader, flipAxes, getTensorList, getTensorList_general, six2mat
import pickle

def test_loader(batch_size):
    for i in range(5):
        yield np.ones([batch_size, 12, 12, 12, 6])


#arr = np.array([0,1,2,3,4,5])
#t=six2mat(arr)
#print("done!")





tensorDir = '/black/localhome/reith/Desktop/projects/Tensors/wh_tensors/'
# 33:57, 32:56, 30:54
crop = [slice(5,77), slice(20,92), slice(2,74)]
crop = [slice(7, 55), slice(40, 88), slice(14, 62)]
resizeFactor = 4
tensors, names = getTensorList_general(tensorDir, giveNames=True, resizeFactor=resizeFactor, crop=crop)

pickle.dump([tensors,names], open('tensorsAndNames48to12.p', 'wb'))

