import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.data.preprocess import resizeTensors, normalizeByIndividualMean, getTensorList_general, cropBlockResize
from src.models.simple_net import SimpleNet





tensorDir = '/black/localhome/reith/Desktop/projects/Tensors/test/'
networkWeights = 'models/trained_simplenet.torch'
wantedShape = (81, 106, 76, 6)
crop = (slice(7, 55), slice(40, 88), slice(14, 62))
resizeFactor = 4

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
predictionStringArr = ["no axis", "the x axis", "the y axis", "the z axis"]
for i, name in enumerate(names):
    print(f"The magic blackbox thinks that {predictionStringArr[prediction[i]]} is flipped for {names[i]}.")
    print(f"[Pseudo certainty is at {F.softmax(net_out[i], dim=0)[prediction[i]].detach().numpy()*100}%]")
print("done")
