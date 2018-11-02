from dipy.io import read_bvals_bvecs
import nibabel as nib

fpath = '/home/reith/PycharmProjects/axisFlipDetector/data/nii8voxDWI'
pbvecs = fpath + '.bvecs'
pbvals = fpath + '.bvals'
ptensors = fpath + ".nii.gz"

print(pbvals, pbvals, ptensors)

bvals, bvecs = read_bvals_bvecs(pbvals, pbvecs)
tensors = arr = nib.load(ptensors).get_fdata().squeeze()
