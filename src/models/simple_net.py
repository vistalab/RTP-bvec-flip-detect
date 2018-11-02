import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, dimIn=12*12*12*6, dimOut=4):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(dimIn, 200)
        self.fc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
