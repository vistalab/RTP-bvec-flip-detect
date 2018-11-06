from detector.simple_net import SimpleNet
from detector.predictFromArchive import predictFromArchive
import torch
from glob import glob
import os

networkWeights = 'detector/trained_simplenet.torch'
wantedShape = (41, 53, 38, 6)
crop = (slice(4, 28), slice(20, 44), slice(7, 31))
resizeFactor = 2

dimIn = 12*12*12*6
dimOut = 4
net = SimpleNet(dimIn=dimIn, dimOut=dimOut).cpu()
net.load_state_dict(torch.load(networkWeights, map_location='cpu'))
archivPath = os.path.dirname(os.path.abspath(__file__))
for archiv in glob(f"{archivPath}/data/dti*.zip"):
    predictFromArchive(archiv, net)
