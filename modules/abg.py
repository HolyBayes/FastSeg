from torch import nn

class AttentionBoostingGate(nn.Module):
    def __init__(self):
        super(AttentionBoostingGate, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(x)
        return x * attention