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
    def forward(self, occupancy_map, flow_map=None):
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
        if flow_map is None:
            return self.convnext.forward_intermediates(occupancy_map_embedding, indices=self.indicies, intermediates_only=True)
        flow_map = einops.rearrange(flow_map, 'b h w t c -> (b t) h w c')
        flow_map_embedding = self.patch_embedding_flow_map(flow_map)
        flow_map_embedding = einops.rearrange(flow_map_embedding, '(b t) h w c -> b c t h w', b=batch_size)
        flow_map_embedding = self.flow_temporal_conv(flow_map_embedding).squeeze(2)
        x = occupancy_map_embedding + flow_map_embedding
        x = self.convnext.forward_intermediates(x, indices=self.indicies, intermediates_only=True)
        
        return x




if __name__ == '__main__':
    pass
