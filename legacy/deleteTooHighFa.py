import pickle
import numpy as np
from skimage.measure import block_reduce


tensorListFa = pickle.load(open("tensorsFullFa.b", "rb"))
tensorList = pickle.load(open("tensorsFull.p", "rb"))

tensorArrFa = np.stack(tensorListFa, axis=0)
tensorArr = np.stack(tensorList, axis=0)

cleanedArr = np.copy(tensorArr)
cleanedArr[tensorArrFa>1] = [0, 0, 0, 0, 0, 0]
tensorList = []

#resize and crop
for t in cleanedArr:
    arr = t
    arr = arr[33:57, 32:56, 30:54]
    # resize to 12x12x12
    # arr = block_reduce(arr, block_size=(2, 2, 2, 1), func=np.mean)
    tensorList.append(arr)

pickle.dump(tensorList, open('tensorsFullnoHighFa_croppednotResized.p', "wb"))
# pickle.dump([t for t in cleanedArr], open('tensorsFullnoHighFa.p', "wb"))
test = np.copy(tensorArrFa)
test[tensorArrFa>1] = 0
print(np.max(test))


