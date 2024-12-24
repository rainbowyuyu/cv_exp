# rainbow_yu cv_exp.cv_unet_pt.unet 🐋✨

import torch


class Conv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU())

    def forward(self, x):
        return self.feature(x)


class UNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        self.conv1 = Conv(in_channel, 64)
        self.conv2 = Conv(64, 128)
        self.conv3 = Conv(128, 256)
        self.conv4 = Conv(256, 512)
        self.conv5 = Conv(512, 1024)
        self.pool = torch.nn.MaxPool2d(2)

        self.up1 = torch.nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6 = Conv(1024, 512)

        self.up2 = torch.nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = Conv(512, 256)

        self.up3 = torch.nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = Conv(256, 128)

        self.up4 = torch.nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = Conv(128, 64)

        self.conv10 = torch.nn.Conv2d(64, out_channel, 3, 1, 1)

    def forward(self, x):
        xc1 = self.conv1(x)
        xp1 = self.pool(xc1)
        xc2 = self.conv2(xp1)
        xp2 = self.pool(xc2)
        xc3 = self.conv3(xp2)
        xp3 = self.pool(xc3)
        xc4 = self.conv4(xp3)
        xp4 = self.pool(xc4)
        xc5 = self.conv5(xp4)

        xu1 = self.up1(xc5)
        xm1 = torch.cat([xc4, xu1], dim=1)
        xc6 = self.conv6(xm1)

        xu2 = self.up2(xc6)
        xm2 = torch.cat([xc3, xu2], dim=1)
        xc7 = self.conv7(xm2)

        xu3 = self.up3(xc7)
        xm3 = torch.cat([xc2, xu3], dim=1)
        xc8 = self.conv8(xm3)

        xu4 = self.up4(xc8)
        xm4 = torch.cat([xc1, xu4], dim=1)
        xc9 = self.conv9(xm4)

        xc10 = self.conv10(xc9)

        return xc10


if __name__ == "__main__":
    input = torch.randn((1, 3, 512, 512))
    # model = Conv(3,3)
    model = UNet(3, 1)
    output = model(input)
    print(output.shape)
