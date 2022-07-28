#Debugging version of discriminator model

import torch
import torch.nn as nn
#from torchsummary import summary


# start with CNN block
class CNNBlock(nn.Module):

    # for convolutional layer
    # stride 2 by default
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        # create a block
        self.conv = nn.Sequential(
            # kernel size 4
            # stride 2 by default
            # bias false as using batch2d
            # padding mode reflect to produce artifact
            nn.Conv2d(in_channels, out_channels,kernel_size=4 , stride=stride, bias=False, padding_mode="reflect"),
            # nn.BatchNorm2d(out_channels), # replaced with instancenorm2d
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),

        )

    # forward propogation
    # input x
    def forward(self, x):
        conv_result = self.conv(x)
        return conv_result


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels=3,  # in channels 3 as default because normally have rgb
                 features=[64, 128, 256, 512],
                 # used in paper: take in channels, send to 64, then 64 to 128, then to 256, and lastly 512
                 ):
        # using this cnn block 4 times
        # send in 256 input, after conv layers get 30 x 30 output
        super().__init__()
        # features are used in cnn block
        # but also have exception in initial block below
        # inside of which have Conv2d and leaky relu
        # no batch norm in initial block in paper
        self.initial = nn.Sequential(
            nn.Conv2d(
                # use in channels *2 because in contrast to normal GANs where send in image and gonna output 0, 1
                # in this case send in satelite image x and send in y as well
                # then concatenate x and y along the channels
                # that's what send in
                # gets the input image and also the output image
                # then based on that it says if the patch of that specific region is fake or real
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        counter =0
        for feature in features[1:]:
            counter = counter+1
            layers.append(
                # specified in paper used stride of 2, on first 3 features
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        # need to output a single value between 0 and 1
        # need another conv layer

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )

        # unpack all of the layers stuff and put that into sequential
        self.model = nn.Sequential(*layers)


    # gonna get x and y as input
    # either a y fake or a y real
    def forward(self, x):

        #x = torch.cat([x, y], dim=1)  # concatentate x and y along the first dimension

        x = self.initial(x)  # send it through the initial thing

        model_x_result = self.model(x)

        return model_x_result


# test case
def test():
    x = torch.randn((1, 3, 256, 256))  # 1 example, 3 channels, 256 x 256 input
    y = torch.randn((1, 3, 256, 256))  # similar for y
    cat = torch.cat([x, y], dim=1)
    model = Discriminator()  # initialize model discriminator
    #print(model)
    preds = model(cat)
    print("self.model(x) shape",preds.shape)
    #summary(model, (x, y))


if __name__ == "__main__":
    test()







