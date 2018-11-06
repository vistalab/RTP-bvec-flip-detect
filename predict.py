import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.data.preprocess import resizeTensors, normalizeByIndividualMean, cropBlockResize, extractNiftiFromZipArchive, getArrayFromNifti
from src.models.simple_net import SimpleNet
import os
from glob import glob


def predictFromArchive(archivPath, net, wantedShape=(41, 53, 38, 6), crop=(slice(4, 28), slice(20, 44), slice(7, 31)), resizeFactor = 2):
    tensorFile = extractNiftiFromZipArchive(archivPath)
    tensor, name = getArrayFromNifti(tensorFile)
    os.remove(tensorFile)
    tensor = [tensor]
    name = [name]
    tensor = resizeTensors(tensor, wantedShape)
    tensor = cropBlockResize(tensor, resizeFactor, crop)
    tensor = normalizeByIndividualMean(tensor)
    tensor = np.stack(tensor)
    tensor = torch.from_numpy(tensor).type(torch.float32)
    tensor = Variable(tensor).view(-1, dimIn)

    net_out = net(tensor)
    prediction = net_out.max(1)[1]

    predictionStringArrOld = ["no axis is flipped", "the x axis is flipped", "the y axis is flipped", "the z axis is flipped", "it has no idea what's happening"]


    predictionStringArrProfessional = ["you can process the data as is",
                                       "you should flip the x axis in the bvec",
                                       "you should flip the y axis in the bvec",
                                       "you should flip the z axis in the bvec",
                                       "You should check this subject manually"]
    for i, name in enumerate([name]):
        predCertainty = F.softmax(net_out[i], dim=0)[prediction[i]].detach().numpy()*100
        pred = prediction[i]
        if predCertainty < 99:
            print(f"{predictionStringArrProfessional[4]} for {name[i]}. ({100-predCertainty}% unsure")
        else:
            print(f"I am {predCertainty:.3f}% pseudo sure that {predictionStringArrProfessional[pred]} for {name[i]}.")
        print(f"[Pseudo certainty is at {predCertainty}% for {predictionStringArrOld[pred]}]")


if __name__ == "__main__":
    archivPath = '/black/localhome/reith/Desktop/projects/neuro_detect/data/raw'
    networkWeights = 'models/trained_simplenet.torch'
    wantedShape = (41, 53, 38, 6)
    crop = (slice(4, 28), slice(20, 44), slice(7, 31))
    resizeFactor = 2

    dimIn = 12*12*12*6
    dimOut = 4
    net = SimpleNet(dimIn=dimIn, dimOut=dimOut)
    net.load_state_dict(torch.load(networkWeights))
    for archiv in glob(f"{archivPath}/dti*.zip"):
        predictFromArchive(archiv, net)
