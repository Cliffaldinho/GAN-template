import torch
import torch.nn as nn
from torch.nn import functional as F

# Residual block
# batch norm needs more than 1 batch size
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,stride=1,use_1x1conv=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,num_channels,kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        #print("Residual block start X: ",X.shape)
        Y = self.conv1(X)
        #print("Residual block conv1 X: ",Y.shape)
        Y = self.bn1(Y)
        #print("Residual block bn1 Y: ",Y.shape)
        Y = F.relu(Y)
        #print("Residual block relu Y finish first: ",Y.shape)
        Y = F.relu(self.bn1(self.conv1(X)))
        #print("Residual block after first section Y: ",Y.shape)

        Y = self.conv2(Y)
        #print("Residual block conv2 Y: ",Y.shape)
        Y = self.bn2(Y)
        #print("Residual block bn2 Y finish second: ",Y.shape)
        Y = self.bn2(self.conv2(Y))
        #print("Residual block after second conv Y: ",Y.shape)

        if self.conv3:
            X = self.conv3(X)
            #print("Residual block conv3 X: ",X.shape)
        Y += X
        #print("Residual block after added skip connection Y: ",Y.shape)
        Y = F.relu(Y)
        #print("Residual block after added relu Y: ",Y.shape)
        #print()
        return Y


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.res1 = Residual(in_channels,features),
        self.res2 = Residual(features, features*2),
        self.res3 = Residual(features*2, features*4)

    def forward(self, x, y):
        outx1, outy1 = self.res1(x, y)
        outx2, outy2 = self.res2(outx1,outy1)
        out = self.res3(outx2, outy2)
