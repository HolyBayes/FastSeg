import sys; sys.path.append('../')
import sys; sys.path.append('../modules/')

from modules.encoder import EfficientResNetEncoder
from modules.decoder import EfficientResNetDecoder
from torch import nn
import torch
from transformers import PretrainedConfig, PreTrainedModel


class SERNetConfig(PretrainedConfig):
    model_type = "sernet"
    
    def __init__(self, num_labels=1, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

class SERNet_Former(PreTrainedModel):
    config_class = SERNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = EfficientResNetEncoder()
        self.decoder = EfficientResNetDecoder(n_classes=config.num_labels)

    def forward(self, x):
        skip_connections, x = self.encoder(x)
        logits = self.decoder(skip_connections, x)
        return logits


# Load your trained model
        
if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    config = SERNetConfig(num_labels=2)
    model = SERNet_Former(config).to(device)
    input = torch.randn((1,3,1024,768)).to(device)
    output = model(input)
    assert output.shape == (1,2,1024,768)
    for _ in range(20): # warmup
        output = model(input)
    start_ts = time.time()
    output = model(input)
    print(f'SERNet inference time (ms): {(time.time()-start_ts)*1000:.4f}') # 40 ms
    torch.save(model, 'sernet.ckpt') # 35 Mb
