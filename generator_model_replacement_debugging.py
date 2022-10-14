# Debugging generator model

import torch
import torch.nn as nn


# similar to cnn block
class Block(nn.Module):

    # down = true - encoder, the downward part of unet, while decode, the upward part of unet
    # one activation (leaky relu) for the encoder, one activation (relu) for the decoder
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(

            # kernel size 4, stride 2, padding 1, bias = false, padding mode = reflect
            # do it if downward part
            # if encoder then want to downsample
            # do it in each of the conv layer

            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect') if down  # downward
            #else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),  # upward
            else nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, padding_mode='reflect')
            ),

            # nn.BatchNorm2d(out_channels), # replaced with instancenorm2d
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),

        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        # used 0.5 dropout in first 3 layers in upper part of unet

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        # image in dimensions 256 x 256

        super().__init__()

        # encoder downsample
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),  # out = 64, dim = 128 x 128
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features * 2, down=True, act='leaky',
                           use_dropout='False')  # out = 128, dim = 64 x 64

        self.down2 = Block(features * 2, features * 4, down=True, act='leaky',
                           use_dropout='False')  # out = 256, dim = 32 x 32

        self.down3 = Block(features * 4, features * 8, down=True, act='leaky',
                           use_dropout='False')  # out = 512, dim = 16 x 16

        self.down4 = Block(features * 8, features * 8, down=True, act='leaky',
                           use_dropout='False')  # out = 512, dim = 8 x 8

        self.down5 = Block(features * 8, features * 8, down=True, act='leaky',
                           use_dropout='False')  # out = 512, dim = 4 x 4

        self.down6 = Block(features * 8, features * 8, down=True, act='leaky',
                           use_dropout='False')  # out = 512, dim = 2 x 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode='reflect'),  # out = 512, dim = 1 x 1
            nn.ReLU()
        )


        # decoder upsample
        # add concatenation from downward part
        # *2 because concatenate from downward part
        self.up1 = Block(features * 8, features * 8, down=False, act='relu', use_dropout=True)  # out = 512,

        self.up2 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)  # out = 512,

        self.up3 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)  # out = 512

        self.up4 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=False)  # out = 512

        self.up5 = Block(features * 8 * 2, features * 4, down=False, act='relu', use_dropout=False)  # out = 256

        self.up6 = Block(features * 4 * 2, features * 2, down=False, act='relu', use_dropout=False)  # out = 128

        self.up7 = Block(features * 2 * 2, features, down=False, act='relu', use_dropout=False)  # out = 64

        # in_channels because want it to be where it was originally
        self.final_up = nn.Sequential(
            #nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=1, padding=1),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(features * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # use tanh because want each pixel value to be between -1 and 1. If 0 and 1 sigmoid.
        )


    def forward(self, x):
        print("x shape: ",x.shape)
        d1 = self.initial_down(x)
        print("d1 shape: ",d1.shape)
        print()




        d2 = self.down1(d1)
        print("d2 shape: ", d2.shape)
        print()

        d3 = self.down2(d2)
        print("d3 shape: ", d3.shape)
        print()

        d4 = self.down3(d3)
        print("d4 shape: ", d4.shape)
        print()


        d5 = self.down4(d4)
        print("d5 shape: ", d5.shape)
        print()


        d6 = self.down5(d5)
        print("d6 shape: ", d6.shape)
        print()

        d7 = self.down6(d6)
        print("d7 shape: ", d7.shape)
        print()

        bottleneck = self.bottleneck(d7)
        print("bottleneck shape: ", bottleneck.shape)
        print()

        up1 = self.up1(bottleneck)
        print("up1 shape: ",up1.shape)
        print()


        c1 = torch.cat([up1,d7],1)
        print("concatenated 1 shape: ",c1.shape)
        up2 = self.up2(c1) # take output from previous, and concatenate with mirror element in encoder. concatenate along dim 1
        print("up2 c1 and conved shape: ", up2.shape)
        print()

        c2 = torch.cat([up2,d6],1)
        print("concatenated 2 shape: ",c2.shape)
        up3 = self.up3(c2)
        print("up3 c2 and conved shape: ", up3.shape)
        print()

        c3 = torch.cat([up3, d5], 1)
        print("concatenated 3 shape: ",c3.shape)
        up4 = self.up4(c3)
        print("up4 c3 and conved shape: ", up4.shape)
        print()

        c4 = torch.cat([up4, d4], 1)
        print("concatenated 4 shape: ",c4.shape)
        up5 = self.up5(c4)
        print("up5 c4 and conved shape: ", up5.shape)
        print()

        c5 = torch.cat([up5, d3], 1)
        print("concatenated 5 shape: ",c5.shape)

        up6 = self.up6(c5)
        print("up6 c5 and conved shape: ", up6.shape)
        print()

        c6 = torch.cat([up6, d2], 1)
        print("concatenated 6 shape: ",c6.shape)

        up7 = self.up7(c6)
        print("up7 c6 and conved shape: ", up7.shape)
        print()

        c7 = torch.cat([up7,d1],1)
        print("concatenated 7 shape: ",c7.shape)

        up8 = self.final_up(c7)
        print("up8 (final up) c7 and conved shape: ", up8.shape)
        print()

        return up8


# test case
def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print("final up shape: ",preds.shape)


if __name__ == "__main__":
    test()




