from preprocess import getTensorList_general
from dipy.reconst.dti import fractional_anisotropy
import numpy as np
from skimage.measure import block_reduce
import pickle
from PIL import Image
import multiprocessing


def six2mat(voxel6):
    return voxel6[[0,3,4,3,1,5,4,5,2]].reshape(3,3)


def getFAfromSix(sixArr, axis=-1):
    mat = six2mat(sixArr)
    eig = np.linalg.eig(mat)[0]
    fa = fractional_anisotropy(eig)
    return fa


def parallel_apply_along_axis(func1d, axis, arr):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.starmap(np.apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


tensorDir = '/home/reith/PycharmProjects/axisFlipDetector/data/tensors/'
tensors = pickle.load(open("tensorsFull.p", "rb"))
# faTensor = np.apply_along_axis(getFAfromSix, -1, tensors)
faTensor = parallel_apply_along_axis(getFAfromSix, -1, tensors)
pickle.dump(faTensor, open("tensorsFullFa.b", "wb"))
