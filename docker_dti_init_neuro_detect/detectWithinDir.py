from simple_net import SimpleNet
from predictFromArchive import predictFromArchive
import torch
from glob import glob
import os
import argparse
import dipy.io as io


def writeBvec(bvec, path):
    writeString = "\n".join([" ".join([str(n) for n in vec]) for vec in bvec.transpose()])
    with open(path, 'w') as f:
        f.write(writeString)

parser = argparse.ArgumentParser()
parser.add_argument('-p', default='/flywheel/v0/output')
args = parser.parse_args()
networkWeights = '/detector/trained_simplenet.torch'
wantedShape = (41, 53, 38, 6)
crop = (slice(4, 28), slice(20, 44), slice(7, 31))
resizeFactor = 2

dimIn = 12*12*12*6
dimOut = 4
net = SimpleNet(dimIn=dimIn, dimOut=dimOut).cpu()
net.load_state_dict(torch.load(networkWeights, map_location='cpu'))
archivPath = args.p
flipDict = {1:'X', 2:'Y', 3:'Z'}
for archiv in glob(os.path.join(archivPath, "dti*.zip")):
    flipPreds = predictFromArchive(archiv, os.path.join(archivPath, "neuro-detect_report.txt"), net)
    whereToFlip = list(flipPreds.values())[0]
    parentDir = os.path.dirname(os.path.abspath(archivPath))
    bvecDir = os.path.join(parentDir, "input/bvec")
    bvecPath = glob(bvecDir + '/*.bvecs')[0]
    originalName = os.path.basename(bvecPath)
    originalName = os.path.splitext(originalName)[0]
    if 0 < whereToFlip < 4:
        bvec = io.read_bvals_bvecs(None, bvecPath)[1]
        bvec[:,whereToFlip-1] = -bvec[:,whereToFlip-1]
        outPath = os.path.join(archivPath, f"{originalName}_correctedBy{flipDict[whereToFlip]}.bvecs")
        writeBvec(bvec, outPath)
        print(f"flipped {originalName} by the {flipDict[whereToFlip]} axis.")
        print(f"Wrote {outPath}")
    else:
        print(f"Didn't automatically correct {originalName}")
