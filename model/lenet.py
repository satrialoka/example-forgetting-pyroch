import torch.nn as nn
import torch.nn.functional as F
inputdim = 1


class LeNet(nn.Module):
    """
    http://yann.lecun.com/exdb/lenet/

    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(inputdim, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
       # print("conv1 out = ",out.size())
        out = F.max_pool2d(out, 2)
        #print("maxpol1 out = ",out.size())
        out = F.relu(self.conv2(out))
        #print("conv2 out = ",out.size())
        out = F.max_pool2d(out, 2)
        #print("maxpol2 out = ",out.size())
        out = out.view(out.size(0), -1)
        #print("outview out = ",out.size())
        out = F.relu(self.fc1(out))
        #print("fc1 out = ",out.size())
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        
        return out