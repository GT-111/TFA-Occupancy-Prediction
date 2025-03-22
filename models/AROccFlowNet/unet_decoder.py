import einops
import torch
import torch.nn as nn
from models.AROccFlowNet.positional_encoding import generate_1d_sin_pos_embedding

# A simple ConvGRU cell for a single time-step update.
class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvGRUCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv_z = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_r = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_q = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        # x: (b, channels, H, W), h_prev: (b, hidden_channels, H, W)
        combined = torch.cat([x, h_prev], dim=1)  # (b, input+hidden, H, W)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined_r = torch.cat([x, r * h_prev], dim=1)
        q = torch.tanh(self.conv_q(combined_r))
        h_new = (1 - z) * h_prev + z * q
        return h_new

# A wrapper that applies the ConvGRU cell across the time dimension.
class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvGRU, self).__init__()
        self.cell = ConvGRUCell(input_channels, hidden_channels, kernel_size, padding)

    def forward(self, x):
        """
        x: (b, t, channels, H, W)
        returns: (b, t, hidden_channels, H, W)
        """
        b, t, _, H, W = x.size()
        # Initialize hidden state to zeros
        h_t = torch.zeros(b, self.cell.hidden_channels, H, W, device=x.device)
        outputs = []
        for i in range(t):
            h_t = self.cell(x[:, i], h_t)
            outputs.append(h_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# The Unet-like decoder that uses ConvGRU for feature fusion.
class UNetDecoder(nn.Module):
    """
    U-Net decoder for multi-scale feature fusion with ConvGRU.
    x: (b, t, d, h//16, w//16)
    features: list of skip connections:
        [ (b, d, h//4, w//4), (b, d, h//8, w//8), (b, d, h//16, w//16) ]
    Returns:
    
    """
    def __init__(self, config):
        """
        Args:
            embed_dims: list of channel dimensions for each scale.
                        For example, [C_high, C_mid, C_low] where C_low is used for the bottleneck.
        """
        super(UNetDecoder, self).__init__()
        self.embed_dims = config.embed_dims
        self.num_waypoints = config.num_waypoints

        self.num_scales = len(self.embed_dims) - 1 # number of upsampling steps
        # We'll use bilinear upsampling (could also use nn.ConvTranspose2d)
        self.upsample_list = nn.ModuleList(nn.Upsample(scale_factor=(1,2,2)) for _ in range(self.num_scales))
        # self.conv_list = nn.ModuleList(nn.Sequential(nn.Conv3d(self.embed_dims[-2-i], self.embed_dims[-2-i], kernel_size=(self.num_waypoints, 1, 1), padding='same'), nn.ELU()) 
                                    #    for i in range(self.num_scales))
        
        # Create a ConvGRU module at each scale.
        # The idea is to fuse the upsampled decoder feature with the corresponding skip connection.
        self.conv_grus = nn.ModuleList()
        for i in range(self.num_scales):
            # At the first upsample step (i=0), the decoder input x comes from the bottleneck
            # with channels=embed_dims[-1] and we fuse it with the lowest resolution skip feature (channels=embed_dims[-2]).
            if i == 0:
                in_channels = self.embed_dims[-1] + self.embed_dims[-2]
                hidden_channels = self.embed_dims[-2]
            else:
                # For subsequent steps, the previous ConvGRU output has hidden_channels equal to the skip connection channels.
                in_channels = self.conv_grus[i-1].cell.hidden_channels + self.embed_dims[-2-i]
                hidden_channels = self.embed_dims[-1-i]
            self.conv_grus.append(ConvGRU(in_channels, hidden_channels, kernel_size=3, padding=1))# self.conv_grus = nn.ModuleList(nn.Sequential(nn.Conv2d(self.embed_dims[-1-i], self.embed_dims[-2-i], kernel_size=3, padding='same'), nn.ELU()) for i in range(self.num_scales))
        
        
        # Final convolution to map to the desired number of output channels.
        # After all upsampling steps, we expect the spatial resolution to match features[0] and
        # the channel dimension to equal embed_dims[0].
        self.final_upsample1 = nn.Upsample(scale_factor=(1,2,2))
        self.final_conv1 = nn.Sequential(nn.Conv2d(2 * self.embed_dims[0], self.embed_dims[0]//2, kernel_size=3, padding='same'), nn.ELU())
        self.final_upsample2 = nn.Upsample(scale_factor=(1,2,2))

        # self.occluded_occupancy_map_decoder = nn.Conv2d(self.embed_dims[0], 1, kernel_size=3, padding='same')
        self.observed_occupancy_map_decoder = nn.Conv2d(self.embed_dims[0]//2, 1, kernel_size=3, padding='same')
        self.flow_map_decoder = nn.Conv2d(self.embed_dims[0]//2, 2, kernel_size=3, padding='same')
    def forward(self, x, features):
        """
        x: (b, t, d, h//16, w//16) - bottleneck features.
        features: list of skip features [feat_high, feat_mid, feat_low].
        """
        b, t, c, h, w = x.size()
        # Process each upsampling step:
        # We assume that features are ordered from high resolution to low resolution,
        # so we fuse in reverse order (starting from the lowest resolution skip).
        for i in range(self.num_scales):
            # Upsample x: merge batch and time for 2D upsampling.
            x_reshaped = einops.rearrange(x, 'b t c h w -> b c t h w')
            x_upsampled = self.upsample_list[i](x_reshaped)
            # Update spatial dims and reshape back
            x = einops.rearrange(x_upsampled, 'b c t h w -> b t c h w') # CONVGRU
            # x = einops.rearrange(x_upsampled, 'b c t h w -> (b t) c h w') # CONV2D
            
            # x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t) # CONV2D
            # Select corresponding skip feature.
            # For i = 0, we expect the resolution of the skip feature to be h//? matching current h,w.
            # Assuming features[-1] is the lowest resolution and features[0] is the highest.
            skip = features[-2 - i]  # e.g., for i=0: lowest resolution skip.
            
            # Since x is 5D and skip is 4D, add a time dimension to skip.
            skip = einops.repeat(skip, 'b c h w -> b t c h w', t=t)

            x = torch.cat([x, skip], dim=2)
            x = self.conv_grus[i](x)
            # Pass through the ConvGRU module.
            # x = x + einops.rearrange(self.conv_list[i](skip), 'b c t h w -> b t c h w')  # returns (b, t, hidden_channels, h, w)
            
        
        # At this point, x should have spatial dimensions matching features[0] and channel dimension embed_dims[0].
        # Apply final 2D convolution on each time slice.
        
        x_reshaped = einops.rearrange(x, 'b t c h w -> b c t h w')
        x_upsampled = self.final_upsample1(x_reshaped)
        # Update spatial dims and reshape back
        x = einops.rearrange(x_upsampled, 'b c t h w -> (b t) c h w')
        x = self.final_conv1(x)
        x_reshaped = einops.rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t)
        x_upsampled = self.final_upsample2(x_reshaped)
        x = einops.rearrange(x_upsampled, 'b c t h w -> (b t) c h w')
       
        observed_occupancy_map = einops.rearrange(self.observed_occupancy_map_decoder(x), '(b t) c h w -> b h w t c', b=b, t=t)
        # occluded_occupancy_map = einops.rearrange(self.occluded_occupancy_map_decoder(x), '(b t) c h w -> b h w t c', b=b, t=t)
        flow_map = einops.rearrange(self.flow_map_decoder(x), '(b t) c h w -> b h w t c', b=b, t=t)
        
        return observed_occupancy_map, flow_map
    
from configs.utils.config import load_config
# Example usage:
if __name__ == '__main__':
    config = load_config('configs/model_configs/AROccFlowNetS.py')
    model_config = config.models.aroccflownet.unet_decoder
    
    decoder = UNetDecoder(model_config)
    embed_dims = model_config.embed_dims
    b, t = 2, 20
    # x from the bottleneck: (b, t, d, h//16, w//16)
    x = torch.randn(b, t, embed_dims[-1], 6, 32)
    print(x.shape)
    # Skip features (make sure spatial sizes match after upsampling):
    feat_high = torch.randn(b, embed_dims[0], 24, 128)  # highest resolution (h//4, w//4)
    feat_mid  = torch.randn(b, embed_dims[1], 12, 64)  # mid resolution (h//8, w//8)
    feat_low  = torch.randn(b, embed_dims[2], 6, 32)  # lowest resolution (h//16, w//16)
    features = [feat_high, feat_mid, feat_low]

    observed_occupancy_map, flow_map = decoder(x, features)
    print(observed_occupancy_map.shape, flow_map.shape)
