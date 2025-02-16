import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import load_config
from models.AROccFlowNet.EfficientMotionPredictor import MotionPredictor
from models.AROccFlowNet.ConvNeXtEncoder import ConvNeXtUNet
from models.AROccFlowNet.ConvLSTM import ConvLSTM
from models.AROccFlowNet.PositionalEncoding import positional_encoding
from utils.SampleModelInput import SampleModelInput
import torch.utils.checkpoint as checkpoint
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
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True),
            num_layers=self.num_layers
        )
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.flow_decoder = nn.Linear(in_features=self.hidden_dim, out_features=2)
        self.occupancy_decoder = nn.Linear(in_features=self.hidden_dim, out_features=1)
        
    def forward(self, input_data_dic):
        cur_agent_states = input_data_dic['cur/his/agent_states']
        cur_agent_types = input_data_dic['cur/his/agent_types']
        cur_predicted_trajs, cur_predicted_traj_scores, cur_context, cur_agent_embeddings = self.motion_predictor(cur_agent_states, cur_agent_types)
        cur_occupancy_map = input_data_dic['cur/his/occupancy_map']
        cur_flow_map = input_data_dic['cur/his/flow_map']
        # TODO: Add the adjacent scene joint_feature
        cur_joint_feature = self.multi_scale_feature_map_encoder(cur_occupancy_map, cur_flow_map)
        
        batch_size, num_agents, num_motion_mode, num_time_steps, _ = cur_predicted_trajs.size()
        cur_predicted_trajs_with_pe = cur_predicted_trajs + positional_encoding(T=self.num_time_steps, D=2)
        cur_marginal_feature = einops.reduce(
            self.trajs_embedding(cur_predicted_trajs_with_pe), 'b a m t h -> b a m h', 'max'
        )
        cur_predicted_traj_score = einops.rearrange(cur_predicted_traj_scores, 'b a m -> b a m 1')
        cur_fused_feature = torch.matmul(
            (cur_marginal_feature + einops.repeat(cur_agent_embeddings, 'b a h -> b a m h', m=num_motion_mode)).view(batch_size, num_agents, self.hidden_dim, num_motion_mode), cur_predicted_traj_score
        ).view(batch_size, num_agents, self.hidden_dim)
        
        prv_occupancy_feature = cur_joint_feature
        flow_list = []
        occupancy_list = []
        for time_step in range(self.num_time_steps):
            cur_fused_feature_projected = self.projection_list[time_step].forward(cur_fused_feature)
            batch_size,  hidden_dim, feature_height, feature_width = prv_occupancy_feature.size()
            prv_occupancy_feature = prv_occupancy_feature.contiguous().view(batch_size, feature_height*feature_width, hidden_dim)
            # TODO: Add the adjacent scene joint_feature
            cur_occpancy_feature = checkpoint.checkpoint(self.transformer_decoder, prv_occupancy_feature, cur_fused_feature_projected)
            cur_occpancy_feature = cur_occpancy_feature.view(batch_size, feature_height, feature_width, -1) 
            prv_occupancy_feature = cur_occpancy_feature.detach()
            cur_occpancy_feature = F.interpolate(cur_occpancy_feature, scale_factor=4, mode='bilinear', align_corners=False)
            cur_occpancy_feature = cur_occpancy_feature.view(batch_size, self.img_size[0], self.img_size[1], -1)
            cur_flow = self.flow_decoder(cur_occpancy_feature).detach()
            cur_occupancy = self.occupancy_decoder(cur_occpancy_feature).detach()

            flow_list.append(cur_flow)  # Reduce memory usage
            occupancy_list.append(cur_occupancy)
            torch.cuda.empty_cache()

        return flow_list, occupancy_list

if __name__ == '__main__':
    config = load_config("configs/AROccFlowNetS.py")
    model = AROccFlowNet(config.models.aroccflownet)
    input_dic = SampleModelInput().generate_sample_input()
    print(model(input_dic))