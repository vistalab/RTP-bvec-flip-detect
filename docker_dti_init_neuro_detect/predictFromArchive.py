import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from preprocess import resizeTensors, normalizeByIndividualMean, cropBlockResize, extractNiftiFromZipArchive, getArrayFromNifti
import os


def predictFromArchive(archivPath, report_file, net, wantedShape=(41, 53, 38, 6), crop=(slice(4, 28), slice(20, 44), slice(7, 31)), resizeFactor = 2):
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
    tensor = Variable(tensor).view(-1, net.fc1.in_features)

    net_out = net(tensor)
    prediction = net_out.max(1)[1]


    predictionStringArrProfessional = ["you can process the data as is",
                                       "you should flip the x axis in the bvec",
                                       "you should flip the y axis in the bvec",
                                       "you should flip the z axis in the bvec",
                                       "you should check this subject manually"]

    returnVals = {}
    with open(report_file, 'a') as rf:
        for i, name in enumerate([name]):
            predCertainty = F.softmax(net_out[i], dim=0)[prediction[i]].detach().numpy()*100
            pred = prediction[i]
            if predCertainty < 99:
                returnVals[name[i]] = 4
                print(f"{predictionStringArrProfessional[4]} for {name[i]}. ({100-predCertainty}% unsure")
                rf.write(f"{predictionStringArrProfessional[4]} for {name[i]}. ({100-predCertainty}% unsure\n")
            else:
                print(f"{predCertainty:.3f}% pseudo sure that {predictionStringArrProfessional[pred]} for {name[i]}.")
                rf.write(f"{predCertainty:.3f}% pseudo sure that {predictionStringArrProfessional[pred]} for {name[i]}.\n")
                returnVals[name[i]] = pred
    return returnVals
