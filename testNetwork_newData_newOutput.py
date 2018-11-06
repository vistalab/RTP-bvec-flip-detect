import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.data.preprocess import resizeTensors, normalizeByIndividualMean, getTensorList_general, cropBlockResize
from src.models.simple_net import SimpleNet

tensorDir = '/black/localhome/reith/Desktop/projects/Tensors/newData/ok/'
networkWeights = 'models/trained_simplenet.torch'
wantedShape = (41, 53, 38, 6)
crop = (slice(4, 28), slice(20, 44), slice(7, 31))
resizeFactor = 2

dimIn = 12*12*12*6
dimOut = 4
net = SimpleNet(dimIn=dimIn, dimOut=dimOut)
net.load_state_dict(torch.load(networkWeights))

tensors, names = getTensorList_general(tensorDir, giveNames=True)
tensors = resizeTensors(tensors, wantedShape)
tensors = cropBlockResize(tensors, resizeFactor, crop)
tensors = normalizeByIndividualMean(tensors)
tensors = np.stack(tensors)
tensors = torch.from_numpy(tensors).type(torch.float32)
tensors = Variable(tensors).view(-1, dimIn)

net_out = net(tensors)
prediction = net_out.max(1)[1]

predictionStringArrOld = ["no axis is flipped", "the x axis is flipped", "the y axis is flipped", "the z axis is flipped", "it has no idea what's happening"]


predictionStringArrProfessional = ["you can process the data as is",
                                   "you should flip the x axis in the bvec",
                                   "you should flip the y axis in the bvec",
                                   "you should flip the z axis in the bvec",
                                   "You should check this subject manually"]
for i, name in enumerate(names):
    predCertainty = F.softmax(net_out[i], dim=0)[prediction[i]].detach().numpy()*100
    pred = prediction[i]
    if predCertainty < 99:
        print(f"{predictionStringArrProfessional[4]} for {names[i]}.")
    else:
        print(f"I am {predCertainty:.3f}% pseudo sure that {predictionStringArrProfessional[pred]} for {names[i]}.")
    print(f"[Pseudo certainty is at {predCertainty}% for {predictionStringArrOld[pred]}]")
print("done")
