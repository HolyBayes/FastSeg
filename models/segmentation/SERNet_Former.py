import sys; sys.path.append('../../')
import sys; sys.path.append('../../modules/')

from modules.encoder import EfficientResNetEncoder
from modules.decoder import EfficientResNetDecoder
from torch import nn
import torch


class SERNet_Former(nn.Module):
    def __init__(self, n_classes):
        super(SERNet_Former, self).__init__()
        self.encoder = EfficientResNetEncoder()
        self.decoder = EfficientResNetDecoder(n_classes)
    
    def forward(self, x):
        activations, x = self.encoder(x)
        return self.decoder(activations, x)
        
if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = SERNet_Former(n_classes=2).to(device)
    input = torch.randn((1,3,1024,768)).to(device)
    output = model(input)
    assert output.shape == (1,2,1024,768)
    for _ in range(20): # warmup
        output = model(input)
    start_ts = time.time()
    output = model(input)
    print(f'SERNet inference time (ms): {(time.time()-start_ts)*1000:.4f}')
