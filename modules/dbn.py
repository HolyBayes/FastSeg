from torch import nn

class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DilationBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DilationBasedNetwork(nn.Module):
    def __init__(self, channels):
        super(DilationBasedNetwork, self).__init__()
        self.dilated_conv1 = DilationBlock(channels, channels, dilation_rate=12)
        self.dilated_conv2 = DilationBlock(channels, channels, dilation_rate=16)
        self.dilated_conv3 = DilationBlock(channels, channels, dilation_rate=18)

    def forward(self, x):
        x1 = self.dilated_conv1(x)
        x2 = self.dilated_conv2(x)
        x3 = self.dilated_conv3(x)
        return x1 + x2 + x3