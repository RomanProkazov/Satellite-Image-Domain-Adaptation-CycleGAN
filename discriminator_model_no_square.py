import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # Initial layer (halves resolution: 640x360 -> 320x180)
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Intermediate layers
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        
        # Final layer (outputs 1x1 prediction)
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)
        
        # Calculate output shape dynamically
        self._initialize_weights()

    def _initialize_weights(self):
        # Test forward pass to verify output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 360, 640)
            output = self.forward(dummy_input)
            print(f"Discriminator output shape: {output.shape}")

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(x)

def test():
    # Test with 640x360 input (batch_size=5, channels=3)
    x = torch.randn((5, 3, 360, 640))  # Note: (H,W) = (360,640)
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(f"Output shape: {preds.shape}")  # Should be (5,1,1,1)

if __name__ == "__main__":
    test()