import easydict
import torch
from modules.SwinTransformerEncoder import SwinTransformerEncoder
from modules.FlowGuidedMultiHeadSelfAttention import FlowGuidedMultiHeadSelfAttention
from modules.Pyramid3DDecoder import Pyramid3DDecoder
from modules.TrajNet import TrajNetCrossAttention



class OFMPNet(torch.nn.Module):
    def __init__(self,config:easydict.EasyDict):

        super(OFMPNet, self).__init__()
        self.config =  config
        self.encoder = SwinTransformerEncoder(config=config)
        self.trajnet_attn = TrajNetCrossAttention(config=config)
        self.fg_msa = config.model.fg_msa
        if self.fg_msa:
            self.fg_msa_layer = FlowGuidedMultiHeadSelfAttention(config=config)
        self.decoder = Pyramid3DDecoder(config=config)
        self.num_waypoints = config.task_config.num_waypoints
    
    def forward(self,occupancy_map, flow_map, road_map, obs_traj, occ_traj):

        res_list = self.encoder(occupancy_map=occupancy_map,flow_map=flow_map, road_map=road_map)
        q = res_list[-1]
        if self.fg_msa:
            res, pos, ref = self.fg_msa_layer(q)
        
        q = res + q
        B, H, W, D = q.size()
        q = q.reshape(B, H*W, D)
        query = torch.repeat_interleave(torch.unsqueeze(q, dim=1),repeats=self.num_waypoints,axis=1)
        
        obs_value = self.trajnet_attn(query, obs_traj, occ_traj)
        
        res = self.decoder(obs_value, res_list)
        return res



















