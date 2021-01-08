import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120) 
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def lenet():
    return {'backbone': LeNet(), 'dim': 84}