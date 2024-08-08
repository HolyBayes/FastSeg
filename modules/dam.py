# Dual-Attention module from HSNet paper https://ieeexplore.ieee.org/document/10495017
import torch
import torch.nn as nn

class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.gavg_pool(x)  # Shape: (B, C, 1, 1)

class ChannelPooling(nn.Module):
    def __init__(self):
        super(ChannelPooling, self).__init__()

    def forward(self, x):
        out = torch.mean(x, dim=1, keepdim=True)
        return out


class ChannelAttention(nn.Module): # CAM module
    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.attn = nn.Sequential(
            GlobalPooling(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x_attn = self.attn(x)
        out = x + x*x_attn
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=2):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.attn = nn.Sequential(
            ChannelPooling(),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=True),
            nn.PReLU(),
            nn.Conv2d(1, 1, dilation=dilation_rate, padding=3*dilation_rate, kernel_size=7, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x_attn = self.attn(x)
        
        out = x + x*x_attn
        return out

class DAMBlock(nn.Module):
    def __init__(self, in_channels):
        super(DAMBlock, self).__init__()
        
        self.cam = ChannelAttention(in_channels, in_channels//2)
        self.sam = SpatialAttention(in_channels, in_channels - in_channels//2)
        
    def forward(self, x):
        cam_out = self.cam(x)
        sam_out = self.sam(x)
        
        return cam_out, sam_out


class DAM(nn.Module):
    def __init__(self, in_channels, n_dam_blocks=2):
        super(DAM, self).__init__()
        
        self.dam = nn.ModuleList([DAMBlock(in_channels) for _ in range(n_dam_blocks)])
        
        self.conv_cam = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv_sam = nn.Conv2d(in_channels, in_channels-in_channels//2, kernel_size=3, padding=1)

    def forward(self, x):
        input = x
        for dam_block in self.dam:
            cam_out, sam_out = dam_block(input)
            input = torch.cat([cam_out, sam_out], dim=1)

        # Concatenate features from both branches
        final_output = torch.cat([cam_out, sam_out], dim=1)
        return final_output


if __name__ == '__main__':
    # Example usage:
    # input_tensor shape: (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 256, 64, 64)
    dam = DAM(in_channels=256)
    output = dam(input_tensor)
    print(output.shape)  # should be the same as input_tensor shape

