from torch import nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2): # "stride" here is a scaling factor
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x