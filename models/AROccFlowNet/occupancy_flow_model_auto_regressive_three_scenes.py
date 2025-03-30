import einops
import torch
import torch.nn as nn
from models.AROccFlowNet.occupancy_flow_model_one_step_three_scene import AROccFlowNetOneStepContext
from models.AROccFlowNet.conv_gru import ConvGRU

class AutoRegWrapperContext(nn.Module):

    def __init__(self, config):
        super().__init__()
        models_config = config
        self.backbone = AROccFlowNetOneStepContext(models_config.backbone_model_config)
        self._load_pretrained_backbone(models_config.pretrained_backbone_path)
        self._freeze_backbone()
        self.embed_dims = self.backbone.embed_dims
        self.hidden_dim = self.backbone.hidden_dim
        self.conv_gru = ConvGRU(models_config.memory_gru)
        self.hidden_channels = models_config.memory_gru.hidden_channels

        self.occupancy_map_decoder = nn.Sequential(nn.Conv2d(self.embed_dims[0] + self.hidden_channels[-1], self.embed_dims[0]//2, kernel_size=1),
                                                   nn.Upsample(scale_factor=(2,2)),
                                                   nn.Conv2d(self.embed_dims[0]//2, 1, kernel_size=3, padding=1))
        self.num_waypoints = models_config.num_waypoints
        # self.memory_lookup = MemoryModule(d_model=self.embed_dims[0], n_heads=4, n_layers=1, memory_size=3)
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _load_pretrained_backbone(self, pretrained_backbone_path):
        if pretrained_backbone_path is None:
            return
        checkpoint = torch.load(pretrained_backbone_path, map_location="cpu")
        ddp_state_dict = checkpoint["model_state_dict"]  # or whatever key you used
        # If your keys have "module." prefix, strip it:
        new_state_dict = {}
        for k, v in ddp_state_dict.items():
            new_k = k.replace("module.", "")
            new_state_dict[new_k] = v

        self.backbone.load_state_dict(new_state_dict, strict=True)


    def forward(self, prv_occupancy_map, cur_occupancy_map, nxt_occupancy_map, gt_prv_occupancy_map, gt_nxt__occupancy_map, training=False):
        # //TODO: Modity the input to take his_occupancy_map, his_flow_map, his_observed_agent_features, his_valid_mask, agent_types
        # device = his_occupancy_map.device
        # batch_size, height, width, _, _ = cur_occpancy_map.shape
        h_hidden = None
        occupancy_map_list = []
        
        for timestep in range(self.num_waypoints):
            with torch.no_grad():
                fused_features = self.backbone.forward(prv_occupancy_map, cur_occupancy_map, nxt_occupancy_map, features_only=True)

            h_hidden = self.conv_gru.forward(fused_features, h_hidden)
            # updated_fused_features = self.memory_lookup.forward(fused_features)

            # res_features = fused_features + updated_fused_features + h_hidden[-1]
            # res_features = torch.cat([fused_features, updated_fused_features, h_hidden[-1]], dim=1)
            occupancy_map = einops.rearrange(self.occupancy_map_decoder(torch.cat([fused_features, h_hidden[-1]], dim=1)), 'b c h w -> b h w 1 c')
            # Teacher forcing
            
                
            occupancy_map_list.append(occupancy_map)
            cur_occupancy_map = torch.cat([cur_occupancy_map[..., 1:, :], occupancy_map], dim=-2)
            prv_occupancy_map = torch.cat([prv_occupancy_map[..., 1:, :], gt_prv_occupancy_map[..., timestep:timestep+1, :]], dim=-2)
            nxt_occupancy_map = torch.cat([nxt_occupancy_map[..., 1:, :], gt_nxt__occupancy_map[..., timestep:timestep+1, :]], dim=-2)

        occupancy_map_res = torch.cat(occupancy_map_list, dim=-2)

        return occupancy_map_res


if __name__ == '__main__':
    pass