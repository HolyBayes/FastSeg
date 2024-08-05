import torch
import torch.nn as nn
from abg import AttentionBoostingGate
from dbn import DilationBasedNetwork


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_abg=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(inplace=True)
        self.with_abg = with_abg
        if with_abg:
            self.abg = AttentionBoostingGate()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        res = self.relu(x)
        if self.with_abg:
            res += self.abg(x)
        return res

class EfficientResNetEncoder(nn.Module):
    def __init__(self):
        super(EfficientResNetEncoder, self).__init__()
        self.initial_conv = ConvBlock(3, 64, stride=2, with_abg=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = ConvBlock(64, 64, stride=1, with_abg=True)
        self.conv2 = ConvBlock(64, 128, stride=2, with_abg=True)
        self.conv3 = ConvBlock(128, 256, stride=2, with_abg=True)
        self.conv4 = ConvBlock(256, 512, stride=1, with_abg=True)
        self.dbn = DilationBasedNetwork(512)

    def forward(self, x):
        activations = []
        for layer in [self.initial_conv, self.maxpool, self.conv1, self.conv2, self.conv3, self.conv4, self.dbn]:
            x = layer(x)
            activations.append(x)
        return activations, x

# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EfficientResNetEncoder().to(device)
    encoder.eval()
    input_tensor = torch.randn(1, 3, 1024, 768).to(device)  # Example input tensor
    import time

    n_warmup_iters = 1
    n_eval_iters = 1

    for _ in range(n_warmup_iters): # warm up
        activations, output = encoder(input_tensor)
    print(f'Device: {output.device}')

    
    start_ts = time.time()
    for _ in range(n_eval_iters):
        encoder(input_tensor)
    print(f'Encoder inference (ms): {(time.time() - start_ts)*1000/n_eval_iters:.4f}') # 60 ms on CPU (!!!)
    for act in activations:
        print(act.shape)
    print(output.shape)    