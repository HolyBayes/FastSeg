import torch
from torch import nn
import sys; sys.path.append('../../')
from modules.abg import AttentionBoostingGate

    
class ContextAFN(nn.Module): # AfN2, S:4
    def __init__(self, channels:int):
        super(ContextAFN, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.PReLU()
        self.abg1 = AttentionBoostingGate(channels)
        
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.PReLU()
        self.abg2 = AttentionBoostingGate(channels)
        
        
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu3 = nn.PReLU()
        self.abg3 = AttentionBoostingGate(channels)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x) + self.abg1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x) + self.abg2(x)
        
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x) + self.abg3(x)
        
        
        return x

class SpatialAFN(nn.Module): # AfN1, S:1
    def __init__(self, channels:int):
        super(SpatialAFN, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.PReLU()
        self.abg1 = AttentionBoostingGate(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.PReLU()
        self.abg2 = AttentionBoostingGate(channels)
        
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu3 = nn.PReLU()
        self.abg3 = AttentionBoostingGate(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x) + self.abg1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x) + self.abg2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x) + self.abg3(x)
        
        return x
