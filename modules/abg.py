from torch import nn

class AttentionBoostingGate(nn.Module):
    def __init__(self, n_channels):
        super(AttentionBoostingGate, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        # x = self.bn(x)
        attention = self.sigmoid(x)
        return x * attention