import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES']='6'

#warmup
print(torch.__version__)
pt_tensor = torch.Tensor([3,4,5])
print(pt_tensor)
pt_tensor2 = torch.Tensor([1,2,3])
print(pt_tensor2)

print("sum t1+t2 =", pt_tensor.add(pt_tensor2))
print("sum t1+t2 =", pt_tensor + pt_tensor2)

print("random tensor is:", torch.rand(2,3))

#simple autograd calc

y = Variable(torch.ones(2,2)*2, requires_grad=True)
z = 2 * (y * y) + 5 * y
z.backward(torch.ones(2,2))
print(y.grad)


# simple NN



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12*12*12*6, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


net = Net()
learning_rate = 0.01
epochs = 666
log_interval = 6
print(net)

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.MultiLabelMarginLoss()
criterion = nn.BCELoss()


for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(dummy_loader(10)):
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 12*12*12*6)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train epoch: {epoch}, loss is {loss.data.item()}")



