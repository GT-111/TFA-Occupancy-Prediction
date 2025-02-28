import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.utils.config import load_config
from models.AROccFlowNet.efficient_motion_predictor import MotionPredictor
from models.AROccFlowNet.convnext_encoder import ConvNeXtUNet
from models.AROccFlowNet.conv_lstm import ConvLSTM
from models.AROccFlowNet.positional_encoding import positional_encoding
from datasets.I24Motion.utils.generate_test_data import SampleModelInput

class AROccFlowNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        self.num_time_steps = config.num_time_steps
        self.hidden_dim = config.hidden_dim
        self.motion_predictor = MotionPredictor(config.motionpredictor)
        self.multi_scale_feature_map_encoder = ConvNeXtUNet(config.convnextunet)
        # self.coarse_featurte_map_encoder = ConvLSTM(config.convlstm)
        self.trajs_embedding = nn.Linear(in_features=2, out_features=self.hidden_dim)
        self.projection_list = nn.ModuleList([
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim) for _ in range(self.num_time_steps)
        ])
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True, norm_first=True),
            num_layers=self.num_layers
        )
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.flow_decoder = nn.Linear(in_features=self.hidden_dim, out_features=2)
        self.observed_occupancy_decoder = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.occluded_occupancy_decoder = nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, his_occupancy_map, his_flow_map, his_observed_agent_features, agent_types, his_valid_mask):
        # //TODO: Modity the input to take his_occupancy_map, his_flow_map, his_observed_agent_features, his_valid_mask, agent_types
        predicted_trajecotries, predicted_trajecotries_scores, _, agent_embeddings = self.motion_predictor.forward(his_observed_agent_features, agent_types, his_valid_mask)
        cur_occupancy_map = his_occupancy_map
        cur_flow_map = his_flow_map
        # //TODO: Add the adjacent scene joint_feature
        cur_joint_feature = self.multi_scale_feature_map_encoder(cur_occupancy_map, cur_flow_map)
        
        batch_size, num_agents, num_motion_mode, num_time_steps, _ = predicted_trajecotries.size()
        predicted_trajs_with_pe = predicted_trajecotries + positional_encoding(T=self.num_time_steps, D=2).to(predicted_trajecotries.device)
        cur_marginal_feature = einops.reduce(
            self.trajs_embedding(predicted_trajs_with_pe), 'b a m t h -> b a m h', 'max'
        )
        cur_predicted_traj_score = einops.rearrange(predicted_trajecotries_scores, 'b a m -> b a m 1')
        cur_fused_feature = torch.matmul(
            (cur_marginal_feature + einops.repeat(agent_embeddings, 'b a h -> b a m h', m=num_motion_mode)).view(batch_size, num_agents, self.hidden_dim, num_motion_mode), cur_predicted_traj_score
        ).view(batch_size, num_agents, self.hidden_dim)
        
        prv_occupancy_feature = cur_joint_feature
        flow_list = []
        observed_occupancy_list = []
        occluded_occupancy_list = []
        for time_step in range(self.num_time_steps):
            cur_fused_feature_projected = self.projection_list[time_step].forward(cur_fused_feature)
            batch_size,  hidden_dim, feature_height, feature_width = prv_occupancy_feature.size()
            prv_occupancy_feature = einops.rearrange(prv_occupancy_feature, 'b d h w -> b (h w) d', h=feature_height, w=feature_width, d=hidden_dim)
            # TODO: Add the adjacent scene joint_feature
            cur_occupancy_feature = self.transformer_decoder(prv_occupancy_feature, cur_fused_feature_projected)
            cur_occupancy_feature = cur_occupancy_feature.view(batch_size, feature_height, feature_width, -1) 
            cur_occupancy_feature = einops.rearrange(cur_occupancy_feature, 'b h w d -> b d h w', h=feature_height, w=feature_width, d=hidden_dim)
            prv_occupancy_feature = cur_occupancy_feature.clone()
            cur_occupancy_feature = F.interpolate(cur_occupancy_feature, scale_factor=4, mode='bilinear', align_corners=False)
            cur_occupancy_feature = einops.rearrange(cur_occupancy_feature, 'b d h w -> b h w d')
            

            flow_list.append(self.flow_decoder(cur_occupancy_feature))
            observed_occupancy_list.append(self.observed_occupancy_decoder(cur_occupancy_feature))
            occluded_occupancy_list.append(self.occluded_occupancy_decoder(cur_occupancy_feature))

        predicted_flow_maps = torch.stack(flow_list, dim=-2)
        predicted_observed_occupancy_maps = torch.stack(observed_occupancy_list, dim=-2)
        predicted_occluded_occupancy_maps = torch.stack(occluded_occupancy_list, dim=-2)

        predicted_trajecotries = predicted_trajecotries
        predicted_trajecotries_scores = predicted_trajecotries_scores

        return predicted_observed_occupancy_maps, predicted_occluded_occupancy_maps, predicted_flow_maps, predicted_trajecotries, predicted_trajecotries_scores


if __name__ == '__main__':
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = load_config("configs/model_configs/AROccFlowNetS.py")
    dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")
    model = AROccFlowNet(model_config.models.aroccflownet).to(device)
    input_dic = SampleModelInput(dataset_config).generate_sample_input(device)
    print(model(input_dic))