import torch.nn.init as init
from torch import nn
import math

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        # Convolutional layers
        if m.weight.requires_grad:
            if m.kernel_size == (1, 1):
                # Initialize 1x1 convolutions to Xavier uniform
                init.xavier_uniform_(m.weight, gain=1)
            else:
                # Initialize other convolutional layers to Kaiming uniform
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None and m.bias.requires_grad:
            # Initialize biases to zero
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        # Fully connected layers
        if m.weight.requires_grad:
            # Initialize weights to Xavier uniform
            init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None and m.bias.requires_grad:
            # Initialize biases to zero
            init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        # Batch normalization layers
        if m.weight is not None and m.weight.requires_grad:
            # Initialize batch norm weights to 1
            init.ones_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            # Initialize batch norm biases to 0
            init.zeros_(m.bias)

    elif isinstance(m, nn.LSTM):
        # LSTM layers
        for param in m.parameters():
            if len(param.shape) >= 2:
                # Treat the weight matrices as 2D matrices
                init.orthogonal_(param.data)
            else:
                # Treat the bias vectors as 1D tensors
                init.normal_(param.data)

    elif isinstance(m, nn.GRU):
        # GRU layers
        for param in m.parameters():
            if len(param.shape) >= 2:
                # Treat the weight matrices as 2D matrices
                init.orthogonal_(param.data)
            else:
                # Treat the bias vectors as 1D tensors
                init.normal_(param.data)