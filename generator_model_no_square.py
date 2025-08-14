import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features=64, num_residuals=9):
        super().__init__()
        # Initial convolution (preserves spatial dimensions)
        self.initial = nn.Conv2d(
            img_channels,
            num_features,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )
        
        # Downsampling blocks (640x360 -> 320x180 -> 160x90)
        self.down_blocks = nn.ModuleList([
            ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
        ])
        
        # Residual blocks (maintains 160x90)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        
        # Upsampling blocks (160x90 -> 320x180 -> 640x360)
        self.up_blocks = nn.ModuleList([
            ConvBlock(
                num_features*4, num_features*2, 
                down=False, 
                kernel_size=3, stride=2, 
                padding=1, output_padding=1
            ),
            ConvBlock(
                num_features*2, num_features, 
                down=False, 
                kernel_size=3, stride=2, 
                padding=1, output_padding=1
            ),
        ])
        
        # Final output layer
        self.last = nn.Conv2d(
            num_features,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

def test():
    # Test with 640x360 input (batch_size=2, channels=3)
    x = torch.randn((2, 3, 360, 640))  # Note: (H,W) = (360,640)
    gen = Generator(img_channels=3)
    print(f"Output shape: {gen(x).shape}")  # Should be (2,3,360,640)

if __name__ == "__main__":
    test()