import easydict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.MultiHeadAttention import MultiHeadAttention




class TrajEncoder(nn.Module):
    def __init__(self, config):
        super(TrajEncoder, self).__init__()
        self.num_heads = config.model.TrajNetCrossAttention.TrajNet.att_heads
        self.out_dim = config.model.TrajNetCrossAttention.TrajNet.out_dim
        self.vector_feature_dim = config.model.TrajNetCrossAttention.TrajNet.vector_feature_dim
        self.node_feature_dim = config.model.TrajNetCrossAttention.TrajNet.node_feature_dim
        
        self.node_feature = nn.Sequential(nn.Conv1d(self.node_feature_dim, 64, kernel_size=1), nn.ELU())
        self.node_attention = MultiHeadAttention(input_channels=(64,64,64), num_heads=self.num_heads, head_size=64, dropout=0.1, output_size=64*5)
        self.vector_feature = nn.Linear(self.vector_feature_dim, 64, bias=False)
        self.sublayer = nn.Sequential(nn.Linear(384, self.out_dim), nn.ELU())

    def forward(self, inputs, mask):
        mask = mask.to(torch.float32)
        mask = torch.matmul(mask[:, :, np.newaxis], mask[:, np.newaxis, :])
        nodes = self.node_feature(inputs[:, :, :self.node_feature_dim].permute(0,2,1))
        nodes = nodes.permute(0,2,1)
        nodes = self.node_attention(inputs=[nodes, nodes, nodes], mask=mask)
        nodes, _ = torch.max(nodes, 1)
        vector = self.vector_feature(inputs[:, 0, self.node_feature_dim:self.node_feature_dim+self.vector_feature_dim])
        out = torch.concat([nodes, vector], dim=1)
        polyline_feature = self.sublayer(out)

        return polyline_feature


    
class Cross_Attention(nn.Module):
    def __init__(self, num_heads, key_dim, conv_attn=False):
        super(Cross_Attention, self).__init__()
        self.mha = MultiHeadAttention(input_channels=(key_dim, key_dim), num_heads=num_heads, head_size=key_dim//num_heads, output_size=key_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(eps=1e-3, normalized_shape=key_dim)
        self.norm2 = nn.LayerNorm(eps=1e-3, normalized_shape=key_dim)
        self.FFN1 = nn.Sequential(nn.Linear(key_dim, 4*key_dim),nn.ELU())
        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Sequential(nn.Linear(4*key_dim, key_dim),nn.ELU())
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, query, key, mask=None, training=True):
        value = self.mha(inputs=[query, key], mask=mask)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value)
        value = self.FFN2(value)
        value = self.dropout2(value)
        value = self.norm2(value)
        return value



class TrajNet(nn.Module):
    def __init__(self,config):
        super(TrajNet, self).__init__()

        self.double_net = config.model.TrajNetCrossAttention.TrajNet.double_net

        
        self.traj_encoder = TrajEncoder(config)
        # self.traj_encoder  = TrajEncoderLSTM(cfg['out_dim'])
        self.no_attn = config.model.TrajNetCrossAttention.TrajNet.no_attention

        self.out_dim = config.model.TrajNetCrossAttention.TrajNet.out_dim
        self.att_heads = config.model.TrajNetCrossAttention.TrajNet.att_heads

        if not self.no_attn:
            if self.double_net:
                self.cross_attention = nn.Module([Cross_AttentionT(num_heads=self.att_heads,key_dim=192,output_dim= self.out_dim) for _ in range(2)])
            else:
                self.cross_attention = Cross_Attention(num_heads=self.att_heads, key_dim= self.out_dim)

        self.obs_norm = nn.LayerNorm(eps=1e-3, normalized_shape= self.out_dim)
        self.occ_norm = nn.LayerNorm(eps=1e-3, normalized_shape= self.out_dim)
        # self.obs_drop = tf.keras.layers.Dropout(0.1)
        # self.occ_drop = tf.keras.layers.Dropout(0.1)

        # dummy_obs_actors = torch.zeros([2,obs_actors,past_to_current_steps,8])
        # dummy_occ_actors = torch.zeros([2,occ_actors,past_to_current_steps,8])
        # dummy_ccl = tf.zeros([1,256,10,7])

        
        self.seg_embed = nn.Linear(2,  self.out_dim, bias=False)

        # self(dummy_obs_actors,dummy_occ_actors)
        # summary(self)
    
    def forward(self,obs_traj,occ_traj,map_traj=None,training=True):

        obs_actors = obs_traj.size()[1]
        occ_actors = occ_traj.size()[1]
        # obs_traj B N T C
        obs_mask = torch.not_equal(obs_traj, 0)[:,:,:,0]
        obs = [self.traj_encoder(obs_traj[:, i],obs_mask[:,i]) for i in range(obs_actors)]
        obs = torch.stack(obs,dim=1)

        occ_mask = torch.not_equal(occ_traj, 0)[:,:,:,0]
        occ = [self.traj_encoder(occ_traj[:, i],occ_mask[:,i]) for i in range(occ_actors)]
        occ = torch.stack(occ,dim=1)
        bi_embed = torch.tensor([[1,0],[0,1]], dtype=torch.float32).repeat_interleave(torch.tensor([obs_actors, occ_actors]), dim=0)
        embed = bi_embed[np.newaxis, :, :].repeat_interleave(occ.size()[0], dim=0).to(obs_traj.device)
        embed = self.seg_embed(embed)

        c_attn_mask = torch.not_equal(torch.max(torch.concat([obs_mask,occ_mask], dim=1).to(torch.int32),dim=-1)[0],0) #[batch,64] (last step denote the current)
        c_attn_mask = c_attn_mask.to(torch.float32)

        if self.no_attn:
            if self.double_net:
                concat_actors = torch.concat([obs,occ], dim=1)
                obs = self.obs_norm(concat_actors+embed)
                occ = self.occ_norm(concat_actors+embed)
                return obs,occ,c_attn_mask
            else:
                return self.obs_norm(obs + embed[:,:obs_actors,:]),self.occ_norm(occ + embed[:,obs_actors:,:]),c_attn_mask

        # interactions given seg_embedding
        concat_actors = torch.concat([obs,occ], dim=1)
        concat_actors = torch.multiply(c_attn_mask[:, :, np.newaxis].to(torch.float32), concat_actors)
        query = concat_actors + embed

        attn_mask = torch.matmul(c_attn_mask[:, :, np.newaxis], c_attn_mask[:, np.newaxis, :]) #[batch,64,64]

        if self.double_net:
            value = self.cross_attention[0](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs,val_occ = value[:,:obs_actors,:] , value[:,obs_actors:,:]

            value_flow = self.cross_attention[1](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs_f,val_occ_f = value_flow[:,:obs_actors,:] , value_flow[:,obs_actors:,:]

            obs = obs + val_obs
            occ = occ + val_occ

            ogm = torch.concat([obs,occ], dim=1) + embed

            obs_f = obs + val_obs_f
            occ_f = occ + val_occ_f

            flow = torch.concat([obs_f,occ_f], dim=1) + embed

            return self.obs_norm(ogm) , self.occ_norm(flow) , c_attn_mask
        
        value = self.cross_attention(query=query, key=concat_actors, mask=attn_mask, training=training)
        val_obs,val_occ = value[:,:obs_actors,:] , value[:,obs_actors:,:]

        obs = obs + val_obs
        occ = occ + val_occ

        concat_actors = torch.concat([obs,occ], dim=1)

        obs = self.obs_norm(obs + embed[:,:obs_actors,:])
        occ = self.occ_norm(occ + embed[:,obs_actors:,:])

        return obs,occ,c_attn_mask

class Cross_AttentionT(nn.Module):
    def __init__(self, num_heads, key_dim,output_dim,conv_attn=False,sep_actors=False):
        super(Cross_AttentionT, self).__init__()
        self.mha = MultiHeadAttention(input_channels=(key_dim * num_heads,key_dim * num_heads), num_heads=num_heads, head_size=key_dim//num_heads,output_size=key_dim,dropout=0.1)
        self.sep_actors = sep_actors
        
        self.norm1 = nn.LayerNorm(eps=1e3, normalized_shape=key_dim)
        self.norm2 = nn.LayerNorm(eps=1e3, normalized_shape=output_dim)
        self.FFN1 = nn.Sequential(nn.Linear(key_dim, 4*key_dim), nn.ELU())
        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Linear(4 * key_dim, output_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.conv_attn = conv_attn

    def forward(self, query, key, mask, training=True,actor_mask=None):
        value = self.mha(inputs=[query, key], mask=mask)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value)
        value = self.FFN2(value)
        value = self.dropout2(value)
        value = self.norm2(value)
        return value

class TrajNetCrossAttention(nn.Module):
    def __init__(self, config:easydict.EasyDict):
        super(TrajNetCrossAttention, self).__init__()

        self.traj_net = TrajNet(config)
        
        self.H, self.W = config.model.TrajNetCrossAttention.pic_size
        self.pic_dim = config.model.TrajNetCrossAttention.pic_dim

        self.sep_actors = config.model.TrajNetCrossAttention.sep_actors
        
        self.num_waypoints = config.task_config.num_waypoints
        self.cross_attn_obs = nn.ModuleList([Cross_AttentionT(num_heads=3, output_dim=self.pic_dim,key_dim=128,sep_actors=self.sep_actors) for _ in range(self.num_waypoints)])




    def forward(self,pic_encode,obs_traj,occ_traj,map_traj=None,training=True):

        obs,occ,traj_mask = self.traj_net(obs_traj,occ_traj,map_traj,training)

        if self.sep_actors:
            actor_mask = torch.matmul(traj_mask[:, :, np.newaxis], traj_mask[:, np.newaxis, :])
        
        flat_encode = torch.reshape(pic_encode, shape=[-1,self.num_waypoints,self.H*self.W,self.pic_dim])
        pic_mask = torch.ones_like(flat_encode[:,0,:,0],dtype=torch.float32)

        obs_attn_mask = torch.matmul(pic_mask[:, :, np.newaxis], traj_mask[:, np.newaxis, :])

        query = flat_encode
        key = torch.concat([obs,occ], dim=1)
        res_list = []
        for i in range(self.num_waypoints):
            if self.sep_actors:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training,actor_mask)
            else:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training)
            v = o + query[:,i]
            res_list.append(v)
            
        obs_value = torch.stack(res_list,dim=1)
        obs_value = torch.reshape(obs_value, shape=[-1,self.num_waypoints,self.H,self.W,self.pic_dim])

        return obs_value





from utils.file_utils import get_config
if __name__=='__main__':
    config = get_config('./config.yaml')

    
    model = TrajNetCrossAttention(config)
    obs_traj = torch.zeros([2,40,40,16])
    occ_traj = torch.zeros([2,12,40,16])
    query = torch.zeros([2, 12, 256, 384])
    res = model(query,obs_traj,occ_traj)
    print(res.shape)