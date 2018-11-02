import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
# os.environ['CUDA_VISIBLE_DEVICES']='4,5,6'
import pickle
import numpy as np
import nibabel as nib

from preprocess import dataLoader, flipAxes, getTensorList, dataLoaderCuda

# Dummy data loader
def dummy_loader(batch_size):
    for i in range(10):
        test = torch.zeros([batch_size, 3])
        test[:] = torch.tensor([1, 0, 1])
        yield torch.rand([batch_size, 12, 12, 12, 6]), test


# NN object
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv3d1 = nn.Conv3d(6, 6, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3d2 = nn.Conv3d(6, 1, kernel_size=(3,3,3), stride=3)
        self.fc1 = nn.Linear(24*24*24, 200)
        self.fc2 = nn.Linear(200, 4)

    def forward(self, x):
        x = F.relu(self.conv3d1(x))
        x = F.relu(self.conv3d2(x))
        x = x.view(-1, 24*24*24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def trainTest(epochs):
    for epoch in range(epochs):
        epochAcc = []
        for batch_idx, (data, target, batchNames) in enumerate(dataLoaderCuda(train, batchSize, randomAxis=True, shuffle=False, names=names)):
            data, target = Variable(data), Variable(target)
            data, target = data, target.cuda()
            #data = data.view(-1, 24*24*24*6)
            data = data.permute(0,4,1,2,3)
            optimizer.zero_grad()
            net_out = net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            #print(f"Loss is {loss}, accuracy is {np.mean((prediction == target).numpy())}")
            loss.backward()
            optimizer.step()
            currAcc = (prediction == target).cpu().numpy()
            if not sum(currAcc) == len(target):
                print(prediction.cpu().numpy()[currAcc == 0])
                print(target.cpu().numpy()[currAcc==0])
                print('\n'.join([batchNames[i] for i in np.where(currAcc == 0)[0]]))
            epochAcc.extend(list(currAcc))
            # if batch_idx % log_interval == 0:
        print(f"\nTrain epoch: {epoch}, loss is {loss.data.item()}, accuracy is {np.mean(epochAcc)}\n")
        allAccuracy =[]
        allWrongs = []
        allWrongNames = []
        for batch_idx, (data, target, batchNames) in enumerate(dataLoaderCuda(test, batchSize, randomAxis=True, shuffle=False, names=names)):
            data_temp = np.copy(data)
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            #data = data.view(-1, 24*24*24*6)
            data = data.permute(0,4,1,2,3)
            net_out = net(data)
            prediction = net_out.max(1)[1]
            selector = (prediction != target).cpu().numpy().astype(np.bool)
            wrongs = data_temp[selector]
            wrongNames = [batchNames[i] for i,j in enumerate(selector) if j]
            for wrong, n in zip(wrongs, wrongNames):
                pass
                # print(n)
            allAccuracy.extend(list((prediction == target).cpu().numpy()))
            allWrongs.extend(wrongs)
            allWrongNames.extend(wrongNames)
            #print(f"Test accuracy is {np.mean(allAccuracy)}")
        print(f"Test accuracy is {np.mean(allAccuracy)}")
        print("Wrong tensors:")
        for n in allWrongNames:
            print(n)



tensorDir = '/home/reith/PycharmProjects/axisFlipDetector/data/tensors/'
tensorDir = '/black/localhome/reith/Desktop/projects/Tensors/wh_tensors/'
#tensors = getTensorList(tensorDir)
tensors, names = pickle.load(open("tensorsAndNames48to24.p", "rb"))
tensors, names = pickle.load(open("tensorsAndNames72to24.p", "rb"))
tensors, names = pickle.load(open("tensorsAndNames72.p", "rb"))



# Create relevant variables
net = Net()
net.cuda()
print(net)
learning_rate = 0.0001
epochs = 1000
log_interval = 1
batchSize = 20
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# criterion = nn.BCELoss()
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.NLLLoss()
test = tensors[:int(len(tensors)*0.2)]
train = tensors[int(len(tensors)*0.2):]

# Train the network
for epoch in range(epochs):
    epochAcc = []
    for batch_idx, (data, target, batchNames) in enumerate(dataLoaderCuda(train, batchSize, randomAxis=True, shuffle=False, names=names)):
        data, target = Variable(data), Variable(target)
        data, target = data, target.cuda()
        #data = data.view(-1, 24*24*24*6)
        data = data.permute(0,4,1,2,3)
        optimizer.zero_grad()
        net_out = net(data)
        prediction = net_out.max(1)[1]
        loss = criterion(net_out, target)
        #print(f"Loss is {loss}, accuracy is {np.mean((prediction == target).numpy())}")
        loss.backward()
        optimizer.step()
        currAcc = (prediction == target).cpu().numpy()
        if not sum(currAcc) == len(target):
            print(prediction.cpu().numpy()[currAcc == 0])
            print(target.cpu().numpy()[currAcc==0])
            print('\n'.join([batchNames[i] for i in np.where(currAcc == 0)[0]]))
        epochAcc.extend(list(currAcc))
        # if batch_idx % log_interval == 0:
    print(f"\nTrain epoch: {epoch}, loss is {loss.data.item()}, accuracy is {np.mean(epochAcc)}\n")


# test the network
allAccuracy =[]
allWrongs = []
allWrongNames = []
for batch_idx, (data, target, batchNames) in enumerate(dataLoaderCuda(tensors, batchSize, randomAxis=True, shuffle=False, names=names)):
    data_temp = np.copy(data)
    data, target = Variable(data), Variable(target)
    data, target = data.cuda(), target.cuda()
    #data = data.view(-1, 24*24*24*6)
    data = data.permute(0,4,1,2,3)
    net_out = net(data)
    prediction = net_out.max(1)[1]
    selector = (prediction != target).cpu().numpy().astype(np.bool)
    wrongs = data_temp[selector]
    wrongNames = [batchNames[i] for i,j in enumerate(selector) if j]
    for wrong, n in zip(wrongs, wrongNames):
        pass
        # print(n)
    allAccuracy.extend(list((prediction == target).cpu().numpy()))
    allWrongs.extend(wrongs)
    allWrongNames.extend(wrongNames)
    #print(f"Test accuracy is {np.mean(allAccuracy)}")

print(f"Test accuracy is {np.mean(allAccuracy)}")
print("Wrong tensors:")
for n in allWrongNames:
    print(n)

print('\nDone!')
