import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import numpy as np
import nibabel as nib
from random import shuffle

from src.data.data_generators import dataLoader
from src.features.build_features import flipEncodedAxes
from src.models.simple_net import SimpleNet
from src.data.preprocess import normalizeByIndividualMean


def test():
    allAccuracy =[]
    allWrongs = []
    allWrongNames = []
    for batch_idx, (data, target, batchNames) in enumerate(dataLoader(testData, batchSize, names=testNames, simulationFunction=flipEncodedAxes, randomAxis=True, shuffle=False)):
        data_temp = np.copy(data)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        data = data.view(-1, dimIn)
        net_out = Net(data)
        prediction = net_out.max(1)[1]
        selector = (prediction != target).cpu().numpy().astype(np.bool)
        wrongs = data_temp[selector]
        wrongNames = [batchNames[i] for i,j in enumerate(selector) if j]
        testAcc = list((prediction == target).cpu().numpy())
        if not sum(testAcc) == len(target) and False:
            print(prediction.cpu().numpy()[testAcc == 0])
            print(target.cpu().numpy()[testAcc==0])
        allAccuracy.extend(testAcc)
        allWrongs.extend(wrongs)
        allWrongNames.extend(wrongNames)
    print(f"Test accuracy is {np.mean(allAccuracy)}")
    print("Wrong tensors:")
    for n in allWrongNames:
        print(n)


def train():
    for epoch in range(epochs):
        epochAcc = []
        lossArr = []
        for batch_idx, (data, target, batchNames) in enumerate(dataLoader(trainData, batchSize, names=trainNames, simulationFunction=flipEncodedAxes, randomAxis=True, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            data = data.view(-1, dimIn)
            optimizer.zero_grad()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            #print(f"Loss is {loss}, accuracy is {np.mean((prediction == target).numpy())}")
            loss.backward()
            optimizer.step()
            currAcc = (prediction == target).cpu().numpy()
            if not sum(currAcc) == len(target) and False:
                print(prediction.cpu().numpy()[currAcc == 0])
                print(target.cpu().numpy()[currAcc==0])
                print('\n'.join([batchNames[i] for i in np.where(currAcc == 0)[0]]))
            epochAcc.extend(list(currAcc))
            lossArr.append(loss.data.item())
            # if batch_idx % log_interval == 0:
        print(f"Train epoch: {epoch}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
        if epoch % test_interval == 0:
            test()


# relevant variables
dimIn = 12*12*12*3
dimOut = 6
learning_rate = 0.001
epochs = 150
log_interval = 1
test_interval = 10
batchSize = 10
tensorDir = "data/processed/encodeList12.p"

Net = SimpleNet(dimIn=dimIn, dimOut=dimOut)
Net.cuda()
print(Net)
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

tensors, names = pickle.load(open(tensorDir, "rb"))
tensors = normalizeByIndividualMean(tensors)
# shuffle together with names
shuffleConstruct = list(zip(tensors, names))
shuffle(shuffleConstruct)
tensors, names = zip(*shuffleConstruct)
testData = tensors[:int(len(tensors)*0.2)]
testNames = names[:int(len(tensors)*0.2)]
trainData = tensors[int(len(tensors)*0.2):]
trainNames = names[int(len(tensors)*0.2):]

# Train the network
train()

learning_rate = 0.0001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate, amsgrad=True)
epochs = 50

# Train the network more
train()

# torch.save(Net.state_dict(), "trained_simplenet.torch")

print('\nDone!')
