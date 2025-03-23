import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.utils.config import load_config
from datasets.I24Motion.utils.generate_test_data import SampleModelInput
from models.AROccFlowNet.occupancy_flow_model_one_step import AROccFlowNetOneStep
from models.AROccFlowNet.memory_gru import MemoryConvGRU
import random

class AutoRegWrapper(nn.Module):

    def __init__(self, config):
        super().__init__()
        models_config = config
        self.backbone = AROccFlowNetOneStep(models_config.backbone_model_config)
        self._load_pretrained_backbone(models_config.pretrained_backbone_path)
        self._freeze_backbone()
        self.embed_dims = self.backbone.embed_dims
        self.hidden_dim = self.backbone.hidden_dim
        self.conv_gru = MemoryConvGRU(models_config.memory_gru)

        self.occupancy_map_decoder = nn.Sequential(nn.Conv2d(self.embed_dims[0], self.embed_dims[0]//2, kernel_size=1),
                                                   nn.Upsample(scale_factor=(2,2)),
                                                   nn.Conv2d(self.embed_dims[0]//2, 1, kernel_size=3, padding=1))
        self.num_waypoints = models_config.num_waypoints
        self.current_step = 0
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

    def get_teacher_forcing_prob(self):
        fraction = min(self.current_step / self.num_waypoints, 1.0)
        return fraction
    
    def forward(self, his_occupancy_map, grond_truth_occupancy_map=None, training=False):
        # //TODO: Modity the input to take his_occupancy_map, his_flow_map, his_observed_agent_features, his_valid_mask, agent_types
        device = his_occupancy_map.device
        cur_occpancy_map = his_occupancy_map
        batch_size, height, width, _, _ = cur_occpancy_map.shape
        h_hidden = None
        occupancy_map_list = []
        
        for timestep in range(self.num_waypoints):
            tf_prob = self.get_teacher_forcing_prob() if training else 0.0
            with torch.no_grad():
                fused_features = self.backbone.forward(cur_occpancy_map, features_only=True)

            if timestep > 0:
                fused_features, h_hidden = self.conv_gru.forward(fused_features, h_hidden)

            occupancy_map = einops.rearrange(self.occupancy_map_decoder(fused_features), 'b c h w -> b h w 1 c')
            # Teacher forcing
            if training and (grond_truth_occupancy_map is not None):
                if random.random() < tf_prob:
                    gt_frame = grond_truth_occupancy_map[..., timestep:timestep+1, :] # (B,H,W,1,C) or (B,H,W,1) 
                    next_frame = gt_frame
                else:
                    next_frame = occupancy_map
            else:
                next_frame = occupancy_map
            occupancy_map_list.append(occupancy_map)

            cur_occpancy_map = torch.cat([cur_occpancy_map[..., 1:, :], next_frame], dim=-2)
            self.current_step = self.current_step + 1
        occupancy_map_res = torch.cat(occupancy_map_list, dim=-2)

        return occupancy_map_res


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ============= Load Configuration =============
    config = load_config('configs/model_configs/AROccFlowNetAutoRegressive.py')
    model_config = config.models
    model = AutoRegWrapper(model_config.auto_regressive_predictor).to(device)
    dataset_config = config.dataset_config
    sample_data_generator = SampleModelInput(dataset_config)
    test_input = sample_data_generator.generate_sample_input(device=device)
    # ============= Test Forward =============
    his_occupancy_map = test_input['cur/his/occupancy_map']
    ground_truth_occupancy_map = test_input['cur/pred/occupancy_map']
    model.forward(his_occupancy_map, ground_truth_occupancy_map, training=True)