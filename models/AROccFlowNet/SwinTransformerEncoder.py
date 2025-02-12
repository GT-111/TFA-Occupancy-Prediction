import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.convnext import convnext_small  # Use ConvNeXt Base

class ConvNeXtFeatureExtractor(nn.Module):
    """ConvNeXt-based feature extractor returning multi-scale features."""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = convnext_small(pretrained=pretrained, features_only=True)

    def forward(self, x):
        """
        Extracts multi-scale features from ConvNeXt.
        Returns:
        - Stage 1: (B, 128, H/4, W/4)
        - Stage 2: (B, 256, H/8, W/8)
        - Stage 3: (B, 512, H/16, W/16)
        - Stage 4: (B, 1024, H/32, W/32)
        """
        return self.backbone(x)

class UNetDecoder(nn.Module):
    """U-Net decoder for multi-scale feature fusion."""
    def __init__(self, embed_dims, out_channels):
        super(UNetDecoder, self).__init__()
        assert len(embed_dims) >= 2, "embed_dims must have at least two levels for U-Net."

        self.upsample_blocks = nn.ModuleList([
            self._upsample_block(embed_dims[i], embed_dims[i-1])
            for i in range(len(embed_dims)-1, 0, -1)
        ])
        
        self.up_conv = nn.ConvTranspose2d(embed_dims[0], embed_dims[0], kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(embed_dims[0], out_channels, kernel_size=1)  

    def _upsample_block(self, in_channels, out_channels):
        """
        Ensures the upsampled feature map matches the skip connection size.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsample
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),  # Reduce channels
            nn.ReLU(inplace=True)
        )


    def forward(self, features):
        assert len(features) == len(self.upsample_blocks) + 1, (
            f"Expected {len(self.upsample_blocks) + 1} feature maps, but got {len(features)}."
        )

        x = features[-1]  # Start from the deepest feature map

        for i, up_block in enumerate(self.upsample_blocks):
            skip = features[-(i+2)]  # Get corresponding skip connection

            # 🔹 1. First, upsample x
            x = up_block[0](x)  # ConvTranspose2d


            # 🔹 2. Concatenate with the skip connection
            x = torch.cat([x, skip], dim=1)

            # 🔹 3. Apply refinement convolution
            x = up_block[1](x)
        
        # 🔹 4. Apply final convolutio
        # x = self.up_conv(x)
        x = self.final_conv(x)
        return x

class ConvNeXtUNet(nn.Module):
    """ConvNeXt-based multi-scale feature extraction with U-Net decoding."""
    def __init__(self, img_size=(256, 256), in_chans=3, out_channels=32, 
                 embed_dims=[96, 192, 384, 768], temporal_depth=3):
        """
        Args:
            img_size: (H, W) input image size
            in_chans: Number of input channels
            out_channels: Number of channels in the final output (D)
            embed_dims: ConvNeXt feature dimensions at different scales
            temporal_depth: Number of time steps for 3D convolution
        """
        super(ConvNeXtUNet, self).__init__()

        self.temporal_depth = temporal_depth
        self.img_size = img_size

        # Use ConvNeXt Base as the feature extractor
        self.convnext = ConvNeXtFeatureExtractor()

        # U-Net decoder for multi-scale feature fusion
        self.unet_decoder = UNetDecoder(embed_dims, out_channels)

        # Temporal modeling with Conv3D
        self.temporal_conv = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels, 
            kernel_size=(self.temporal_depth, 1, 1), padding=(0, 0, 0), 
            stride=(1, 1, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        assert (H, W) == self.img_size, (
            f"Expected input size ({self.img_size}), but got ({H}, {W})."
        )
        assert T >= self.temporal_depth, (
            f"Temporal depth ({self.temporal_depth}) must not exceed number of frames ({T})."
        )

        x = rearrange(x, "b t c h w -> (b t) c h w")  

        # Extract multi-scale features from ConvNeXt
        features = self.convnext(x)

        # Print extracted feature sizes for verification


        assert len(features) == len(self.unet_decoder.upsample_blocks) + 1, (
            f"Expected {len(self.unet_decoder.upsample_blocks) + 1} feature maps, got {len(features)}."
        )

        fused_features = self.unet_decoder(features)  
        
        fused_features = rearrange(fused_features, "(b t) c h w -> b c t h w", b=B, t=T)  
        
        x = self.temporal_conv(fused_features).squeeze(2)
        
        return x  # Final shape: (B, D, H, W)

if __name__ == '__main__':
    # Initialize model
    model = ConvNeXtUNet(img_size=(256, 256), in_chans=3, out_channels=32, temporal_depth=3)

    # Create a dummy input (B=4, T=3, C=3, H=64, W=80)
    dummy_input = torch.randn(4, 3, 3, 256, 256)

    # Forward pass
    output = model(dummy_input)

    # Verify output shape
    print("✅ Output shape:", output.shape)  # Expected: (4, 3, 64, 80, 32)
