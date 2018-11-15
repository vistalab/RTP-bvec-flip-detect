from simple_net import SimpleNet
from predictFromArchive import predictFromArchive
import torch
from glob import glob
import os
import argparse


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
for archiv in glob(os.path.join(archivPath, "dti*.zip")):
    predictFromArchive(archiv, os.path.join(archivPath, "neuro-detect_report.txt"), net)
