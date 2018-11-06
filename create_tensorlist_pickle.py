from preprocess import dataLoader, flipAxes, getTensorList, dataLoaderCuda, getTensorList_general
import pickle

'''
tensorDir = '/home/reith/PycharmProjects/axisFlipDetector/data/tensors/'
tensorList, names = getTensorList(tensorDir, giveNames=True)

pickle.dump([tensorList, names], open("tensorsAndNamesFull.p", "wb"))
'''


tensorList, names = pickle.load(open("tensorsAndNamesFull.p", "rb"))
resizedList = []
for t in tensorList:
    arr = t
    arr = arr[33:57, 32:56, 30:54]
    # resize to 12x12x12
    # arr = block_reduce(arr, block_size=(2, 2, 2, 1), func=np.mean)
    resizedList.append(arr)

pickle.dump([resizedList, names], open("tensorsAndNamesFullCropped.p", "wb"))
