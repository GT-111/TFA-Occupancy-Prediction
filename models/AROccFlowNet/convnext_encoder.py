from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.convnext import ConvNeXt, convnext_tiny
from configs.utils.config import load_config
import einops

class PatchEmbed(nn.Module):
    def __init__(self, config):
        super(PatchEmbed, self).__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        patches_resolution = [self.img_size[0] //
                              self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim

        self.proj = nn.Conv2d(in_channels=self.in_chans, out_channels=self.embed_dim, kernel_size=self.patch_size,
                           stride=self.patch_size)
        
        self.norm = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-5)
        

    def forward(self, x):
        B, H, W, C = x.size()

        x = x.permute(0,3,1,2) # B C H W
        x = self.proj(x)
        
        x = x.permute(0,2,3,1) # B H W C
        x = torch.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        
        x = self.norm(x)
        x = torch.reshape(x, shape=[B, H // self.patch_size[0], W // self.patch_size[0], self.embed_dim])

        return x

class ConvNeXtFeatureExtractor(nn.Module):
    """ConvNeXt-based feature extractor returning multi-scale features."""
    def __init__(self, config):
        super().__init__()
        
        self.embed_dims = config.embed_dims
        self.depths = config.depths
        self.convnext = ConvNeXt(dims= self.embed_dims, depths=self.depths)
        self.shallow_decode = config.shallow_decode
        self.indicies = [stage for stage in range(len(self.embed_dims) + 1)][1:-self.shallow_decode]
        
        
        patch_embedding_occupancy_map_config = config.patch_embedding_occupancy_map
        patch_embedding_flow_map_config = config.patch_embedding_flow_map
        self.patch_embedding_occupancy_map = PatchEmbed(patch_embedding_occupancy_map_config)
        self.patch_embedding_flow_map = PatchEmbed(patch_embedding_flow_map_config)
        
        self.hidden_dim = config.hidden_dim
        # self.occupancy_conv = nn.Conv2d(in_channels=self.embed_dims[0], out_channels=self.hidden_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

        self.flow_temporal_depth = config.flow_temporal_depth
        self.flow_temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.embed_dims[0], out_channels=self.embed_dims[0], kernel_size=(self.flow_temporal_depth, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
            nn.ReLU(inplace=True)
            )
    def forward(self, occupancy_map, flow_map):
        """
        Extracts multi-scale features from ConvNeXt.
        occupancy_map: (B, H, W, T, C)
        flow_map: (B, H, W, T, C)
        Returns:
        - Stage 1: (B, 128, H/4, W/4)
        - Stage 2: (B, 256, H/8, W/8)
        - Stage 3: (B, 512, H/16, W/16)
        - Stage 4: (B, 1024, H/32, W/32)
        """
        batch_size, height, width, temporal_depth, _ = occupancy_map.size()
        occupancy_map_embedding = self.patch_embedding_occupancy_map(occupancy_map.reshape(batch_size, height, width, temporal_depth))
        occupancy_map_embedding = einops.rearrange(occupancy_map_embedding, 'b h w c -> b c h w')
        # occupancy_map_embedding = self.occupancy_conv(occupancy_map_embedding)
        
        flow_map = einops.rearrange(flow_map, 'b h w t c -> (b t) h w c')
        flow_map_embedding = self.patch_embedding_flow_map(flow_map)
        flow_map_embedding = einops.rearrange(flow_map_embedding, '(b t) h w c -> b c t h w', b=batch_size)
        flow_map_embedding = self.flow_temporal_conv(flow_map_embedding).squeeze(2)
        x = occupancy_map_embedding + flow_map_embedding
        x = self.convnext.forward_intermediates(x, indices=self.indicies, intermediates_only=True)
        
        return x

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


    def forward(self, x, features, pos_embeddings):
        
        
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
    def __init__(self, config):
        """
        Args:
            img_size: (H, W) input image size
            in_chans: Number of input channels
            out_channels: Number of channels in the final output (D)
            embed_dims: ConvNeXt feature dimensions at different scales
            temporal_depth: Number of time steps for 3D convolution
        """
        super(ConvNeXtUNet, self).__init__()

        self.temporal_depth = config.temporal_depth
        self.img_size = config.img_size
        self.embed_dims = config.embed_dims
        self.out_channels = config.out_channels
        # Use ConvNeXt Base as the feature extractor
        self.convnext = ConvNeXtFeatureExtractor()

        # U-Net decoder for multi-scale feature fusion
        self.unet_decoder = UNetDecoder(self.embed_dims, self.out_channels)

        # Temporal modeling with Conv3D
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(self.temporal_depth, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
            nn.ReLU(inplace=True)
            )

        self.up_conv = nn.ConvTranspose2d(self.embed_dims[0], self.embed_dims[0], kernel_size=2, stride=2)
    def forward(self, occupancy_map, flow_map):
        """
        Args:
            occupancy_map: Occupancy map input (B, H, W, T, C)
            flow_map: Flow map input (B, H, W, T, C)

        """
        fearture_map = torch.cat([occupancy_map[..., 1:, :], flow_map], dim=-1)
        fearture_map = einops.rearrange(fearture_map, "b h w t c -> b t c h w")  # (B, T, C, H, W)
        B, T, C, H, W = fearture_map.shape

        assert (H, W) == self.img_size, (
            f"Expected input size ({self.img_size}), but got ({H}, {W})."
        )
        assert T >= self.temporal_depth, (
            f"Temporal depth ({self.temporal_depth}) must not exceed number of frames ({T})."
        )

        fearture_map = einops.rearrange(fearture_map, "b t c h w -> (b t) c h w")  

        # Extract multi-scale features from ConvNeXt
        features = self.convnext(fearture_map)

        # Print extracted feature sizes for verification


        assert len(features) == len(self.unet_decoder.upsample_blocks) + 1, (
            f"Expected {len(self.unet_decoder.upsample_blocks) + 1} feature maps, got {len(features)}."
        )

        fused_features = self.unet_decoder(features)  
        
        fused_features = einops.rearrange(fused_features, "(b t) c h w -> b c t h w", b=B, t=T)  
        
        fused_features = self.temporal_conv(fused_features).squeeze(2)

        
        return fused_features  # Final shape: (B, C, H/4, W/4)

if __name__ == '__main__':
    pass
