import torch
import torch.nn as nn
from torch.nn import functional as F

# Residual block
# batch norm needs more than 1 batch size
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,kernel_size=3,stride=1,padding=1,use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,num_channels,kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None

        self.gn1 = nn.GroupNorm(num_groups=int(num_channels / 16), num_channels=num_channels)
        self.gn2 = nn.GroupNorm(num_groups=int(num_channels / 16), num_channels=num_channels)

    def forward(self, X):
        #print("Residual block start X: ",X.shape)
        Y = self.conv1(X)
        #print("Residual block conv1 X: ",Y.shape)
        Y = self.gn1(Y)
        #print("Residual block bn1 Y: ",Y.shape)
        Y = F.relu(Y)
        #print("Residual block relu Y finish first: ",Y.shape)
        Y = F.relu(self.gn1(self.conv1(X)))
        #print("Residual block after first section Y: ",Y.shape)

        Y = self.conv2(Y)
        #print("Residual block conv2 Y: ",Y.shape)
        Y = self.gn2(Y)
        #print("Residual block bn2 Y finish second: ",Y.shape)
        Y = self.gn2(self.conv2(Y))
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


class Resizer(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, features=64):
        super().__init__()
        #self.res1 = Residual(in_channels,features,use_1x1conv=True)
        self.conv_in = nn.Conv2d(in_channels, features, 3, 1, 1, bias=False, padding_mode='reflect')
        self.res1 = Residual(features, features)
        self.res2 = Residual(features, features)
        self.res3 = Residual(features, features)
        self.res4 = Residual(features,features)
        self.conv_out = nn.Conv2d(features,out_channels, 3, 1, 1, bias=False, padding_mode='reflect')
        #self.res5 = Residual(features, in_channels,use_1x1conv=True)

    def forward(self, x):

        conv_in = self.conv_in(x)
        res1 = self.res1(conv_in)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        conv_out = self.conv_out(res4)

        return conv_out


def test():
    x = torch.randn((1, 6, 1024,1024))
    model = Resizer(in_channels=6, out_channels=3, features=64)
    preds = model(x)
    print("final up shape: ", preds.shape)

if __name__ == "__main__":
    test()