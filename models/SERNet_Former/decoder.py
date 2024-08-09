import torch
from torch import nn
from typing import Literal, List, Optional
import sys; sys.path.append('../')
from modules.afn import ContextAFN, SpatialAFN
from enum import Enum
from models.SERNet_Former.encoder import EfficientResNetEncoder
from modules.upsample import UpsampleBlock




class AfNType(Enum):
    AfN1 = "AfN1" # Spatial, no upsampling
    AfN2 = "AfN2" # Context, upsampling by 4

class AfnDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, afn_type: AfNType):
        super(AfnDecoderBlock, self).__init__()
        if afn_type == AfNType.AfN1:
            self.afn = SpatialAFN(out_channels)
            self.upsample_block1 = UpsampleBlock(in_channels, out_channels, stride=1)  # Use stride for upsampling
            self.upsample_block2 = UpsampleBlock(out_channels, out_channels, stride=1)  # No additional upsampling needed
        elif afn_type == AfNType.AfN2:
            self.afn = ContextAFN(out_channels)
            self.upsample_block1 = UpsampleBlock(in_channels, out_channels, stride=4)  # Use stride for upsampling
            self.upsample_block2 = UpsampleBlock(out_channels, out_channels, stride=4)  # Further upsample after AfN2
        else:
            raise ValueError(f"Unsupported afn_type: {afn_type}")

    def forward(self, x):
        x = self.upsample_block1(x)
        y = self.afn(x)
        y = self.upsample_block2(y)
        return x + y


class EfficientResNetDecoder(nn.Module):
    def __init__(self, n_classes:int, encoder_dims:List[int]=[64, 64, 64, 128, 256, 512]):
        super(EfficientResNetDecoder, self).__init__()
        self.encoder_dims = encoder_dims
        self.afn1 = AfnDecoderBlock(self.encoder_dims[2],64,AfNType.AfN1)
        self.afn2 = AfnDecoderBlock(self.encoder_dims[-1],32, AfNType.AfN2)
        self.conv1 = nn.Conv2d(64+32+self.encoder_dims[2], 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.PReLU()
        self.upsample1 = UpsampleBlock(64, 64, stride=2)
        self.upsample2 = UpsampleBlock(64+self.encoder_dims[0], 128, stride=2)
        self.cls_head = nn.Linear(128, n_classes, bias=True)

    def forward(self, skip_connections, x):
        skip1, _, skip2, _, _, _, skip3 = skip_connections
        afn1 = self.afn1(skip2)
        afn2 = self.afn2(skip3)
        x = torch.cat([skip2, afn1, afn2], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.upsample1(x)
        x = torch.cat([skip1,x], dim=1)
        x = self.upsample2(x)
        x = x.permute(0,2,3,1)
        x = self.cls_head(x)
        x = x.permute(0,3,1,2)
        return x


if __name__ == '__main__':
    import time
    
    block1 = AfnDecoderBlock(64,128,AfNType.AfN1)
    block2 = AfnDecoderBlock(64,128,AfNType.AfN2)
    input = torch.randn((1,64,64,48))
    output1 = block1(input)
    output2 = block2(input)
    assert output1.shape == (1, 128, 64, 48)
    assert output2.shape == (1, 128, 256, 192)

    n_classes=2
    device = torch.device('cpu')

    encoder = EfficientResNetEncoder().to(device)
    decoder = EfficientResNetDecoder(n_classes=n_classes).to(device)
    input_tensor = torch.randn(1, 3, 1024, 768).to(device)  # Example input tensor
    print(f'Device: {input_tensor.device}')
    for _ in range(10): # warmup
        skip_connections, x = encoder(input_tensor)
        output = decoder(skip_connections, x)
    start_ts = time.time()
    skip_connections, x = encoder(input_tensor)
    output = decoder(skip_connections, x)
    print(f'Total time (ms): {1000*(time.time()-start_ts):.4f}')

    assert output.shape == (1,n_classes,1024,768)
    