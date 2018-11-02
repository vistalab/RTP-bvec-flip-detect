import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, color_fa
import numpy as np
import nibabel as nib
from dipy.reconst.dti import fractional_anisotropy
from PIL import Image
import pickle
from preprocess import getTensorList_general, getEncodeArray

def six2mat(voxel6):
    return voxel6[[0, 1, 3, 1, 2, 4, 3, 4, 5]].reshape(3,3)


def getFaArrRaw(t):
    arr = t
    matArr = np.apply_along_axis(six2mat, -1, arr)
    evalArr, evecArr = np.linalg.eig(matArr)
    faArr = fractional_anisotropy(evalArr)
    return [faArr, evalArr, evecArr]





fpath = '/home/reith/PycharmProjects/axisFlipDetector/data/tensors/5aac14517f233f00193eb257_tensors.nii.gz'
outputFolder = '/black/localhome/reith/Desktop/projects/Tensors/color_images/'

tensorPath = '/black/localhome/reith/Desktop/projects/Tensors/wh_tensors/'
crop = [slice(7, 55), slice(40, 88), slice(14, 62)]
# crop = [slice(2, 14), slice(10,22), slice(4, 18)]
tensors, names = getTensorList_general(tensorPath, giveNames=True, crop=crop, resizeFactor=4)

faArrs = []
for i, t in enumerate(tensors):
    faArrs.append(getFaArrRaw(t))
    print(f"{i} of {len(names)} nifty files are processed!")

encodeList = []
for i, ls in enumerate(faArrs):
    fa, eval, evec = ls[0], ls[1], ls[2]
    result = np.copy(eval)
    for j in range(fa.shape[0]):
        for k in range(fa.shape[1]):
            for l in range(fa.shape[2]):
                vector = evec[j,k,l][eval[j,k,l].argmax()]
                result[j,k,l] = vector*fa[j,k,l]
    encodeList.append(result)
    print(f"{i} of {len(names)} nifty files are encoded!")

pickle.dump([encodeList, names], open("encodeList.p", "wb"))
print("done")

