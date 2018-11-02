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


def six2mat(voxel6):
    return voxel6[[0, 1, 3, 1, 2, 4, 3, 4, 5]].reshape(3,3)


def getFaArr(fpath):
    arr = nib.load(fpath).get_fdata().squeeze()
    matArr = np.apply_along_axis(six2mat, -1, arr)
    evalArr, evecArr = np.linalg.eig(matArr)
    faArr = fractional_anisotropy(evalArr)
    return faArr





fpath = '/home/reith/PycharmProjects/axisFlipDetector/data/tensors/5aac14517f233f00193eb257_tensors.nii.gz'
outputFolder = '/black/localhome/reith/Desktop/projects/Tensors/color_images/'

tensorPath = '/black/localhome/reith/Desktop/projects/Tensors/wh_tensors/'
fpaths = [os.path.join(tensorPath , p) for p in os.listdir(tensorPath)]

faArrs = []
for i, fpath in enumerate(fpaths):
    faArrs.append(getFaArr(fpath))
    print(f"{i} of {len(fpaths)} nifty files are processed!")

pickle.dump(faArrs, open("faArrList.p", "wb"))
print("done")











'''
fpath = '/home/reith/PycharmProjects/axisFlipDetector/dwi/'

fdwi =  os.path.join(fpath, 'file.nii.gz')
fbval = os.path.join(fpath, 'file.bval')
fbvec = os.path.join(fpath, 'file.bvec')



data, affine = load_nifti(fdwi)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data)
img = Image.fromarray((test[:,:,30,:]*255).astype(np.uint8))
img.show()
save_nifti('colorfa.nii.gz', tenfit.color_fa, affine)
'''
