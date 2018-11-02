import numpy as np
import pickle
import torch
from src.data.preprocess import getTensorList_general
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.measure import block_reduce
from scipy.ndimage import zoom
from dipy.reconst.dti import fractional_anisotropy



class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.fc1 = nn.Linear(12*12*12*6, 200)
        self.fc2 = nn.Linear(200, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def six2mat(voxel6):
    return voxel6[[0, 1, 3, 1, 2, 4, 3, 4, 5]].reshape(3,3)


def getFaArr(arr):
    matArr = np.apply_along_axis(six2mat, -1, arr)
    evalArr, evecArr = np.linalg.eig(matArr)
    faArr = fractional_anisotropy(evalArr)
    return faArr

def resizeTensors(tensors, wantedShape):
    resizedTensors = []
    count = 0
    oldShapes = []
    newShapes = []
    print("Resizing..")
    for t in tensors:
        if t.shape != wantedShape:
            count += 1
            oldShapes.append(t.shape)
            zoomFactor = []
            for i, s in enumerate(t.shape):
                zoomFactor.append(wantedShape[i]/s)
            t = zoom(t, zoomFactor)
            newShapes.append(t.shape)
        resizedTensors.append(t)
    for i in range(count):
        print(f"{oldShapes[i]} -> {newShapes[i]}")
    print(f"Resized {count} out of {len(tensors)} tensors. (This takes quite long)")
    return resizedTensors

def flipAxes(arr, axis):
    # axes is len 3 array, [1,0,0], is flip x, [1,0,1] means flip x and z and so on
    if axis == 1:
        arr[:, :, :, [1, 3]] = -arr[:, :, :, [1, 3]]
    if axis == 2:
        arr[:, :, :, [1, 4]] = -arr[:, :, :, [1, 4]]
    if axis == 3:
        arr[:, :, :, [3, 4]] = -arr[:, :, :, [3, 4]]
    return arr

def cropBlockResize(tensors, resizeFactor, crop):
    result = []
    rf = resizeFactor
    for t in tensors:
        t = t[crop]
        t = block_reduce(t, block_size=(rf, rf, rf, 1), func=np.mean)
        result.append(t)

    return result

def normalizeByMean(tensors):
    result = []
    for t in tensors:
        t = t/np.mean(t)
        result.append(t)
    print(f"Normalized {len(tensors)} tensors")
    return result

tensorDir = '/black/localhome/reith/Desktop/projects/Tensors/test/'
networkWeights = 'trained_simplenet.torch'
wantedShape = (81, 106, 76, 6)
crop = (slice(7, 55), slice(40, 88), slice(14, 62))
resizeFactor = 4


net = simpleNet()
net.load_state_dict(torch.load(networkWeights))

tensors, names = getTensorList_general(tensorDir, giveNames=True)
# tensors = resizeTensors(tensors, wantedShape)

# for t, n in zip(tensors,names):
#     if n[-11:] == 'oMNI.nii.gz':
#         print(n)
#         print(t[35,35,66])
#         print(getFaArr(t[35,35,66]))

t = tensors[3] # YflipAutoMNI

# One of the highest Fas in the area (0.48)
# for t, n in zip(tensors,names):
#     if n[-11:] == 'oMNI.nii.gz':
#         print(n)
#         print(t[40,50, 42])
#         print(getFaArr(t[40,50, 42]))

# One of the highest Fas in the area (0.44)
rightTensors = []
rightNames = []
for t, n in zip(tensors,names):
    if n[-11:] == 'oMNI.nii.gz':
        rightTensors.append(t)
        rightNames.append(n)
tensors = rightTensors
names = rightNames
xflip = tensors[2]
yflip = tensors[0]
xflipx = np.copy(xflip)
xflipx = flipAxes(xflipx, 1)
xflipy = np.copy(xflip)
xflipy = np.copy(flipAxes(xflipy, 2))
xflipz = np.copy(xflip)
xflipz = np.copy(flipAxes(xflipz, 3))

print('xflip\n', np.linalg.eig(six2mat(xflip[45,55,47])))
print('yflip\n', np.linalg.eig(six2mat(yflip[45,55,47])))
print('xflipx\n', np.linalg.eig(six2mat(xflipx[45,55,47])))
print('xflipy\n', np.linalg.eig(six2mat(xflipy[45,55,47])))
print('xflipz\n', np.linalg.eig(six2mat(xflipz[45,55,47])))





for t, n in zip(tensors,names):
    if n[-11:] == 'oMNI.nii.gz':
        print(n)
        print(t[45,55, 47])
        print(np.linalg.eig(six2mat(t[45,55,47]))[1])
        print(getFaArr(t[45,55, 47]))


print("done!")
