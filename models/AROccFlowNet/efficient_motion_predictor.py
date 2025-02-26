# Custom Imlpementation of the Effcient Muotion Prediction (EMP) module
# Since the I24 has no road map, the lane encoder is not used in this implementation
# For details, please refer to the original paper: https://arxiv.org/abs/2409.16154

import torch
import torch.nn as nn
import einops
from configs.utils.config import load_config

class MotionEncoder(nn.Module):

    def __init__(self, num_states, hidden_dim, num_heads, dropout_prob=0.1, num_layers=1):
        super().__init__()

        self.num_states = num_states
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.num_embeddings = 6
        # 0 : sedan
        # 1 : midsize
        # 2 : van
        # 3 : pickup
        # 4 : semi
        # 5 : truck
        self.agent_projection = nn.Linear(in_features=self.num_states, out_features=hidden_dim)
        self.temporal_attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dropout=self.dropout_prob, batch_first=True), num_layers=self.num_layers)
        self.agent_type__learnable_embeddings = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=hidden_dim)
        self.agent_type__learnable_embeddings.weight.data = nn.init.xavier_uniform_(self.agent_type__learnable_embeddings.weight.data)
        
        #

        
    def forward(self, agent_states, agent_types, valid_mask):
        """
        agent_states: torch.tesor (batch_size, num_agents, num_timesteps, num_states) 
        agent_types: torch.tensor (batch_size, num_agents)
        valid_mask: torch.tensor (batch_size, num_agents, num_timesteps)
        """
        batch_size, num_agents, num_timesteps, num_states = agent_states.size()
        agent_embeddings = self.agent_projection(agent_states) # (batch_size, num_agents, num_timesteps, hidden_dim)
        agent_embeddings = agent_embeddings * valid_mask.unsqueeze(-1)
        agent_embeddings = einops.rearrange(agent_embeddings, 'b a t h -> (b a) t h')

        agent_embeddings = self.temporal_attention.forward(agent_embeddings)
        agent_embeddings = einops.reduce(agent_embeddings, 'b t h -> b h', 'max')
        agent_embeddings = einops.rearrange(agent_embeddings, '(b a) h -> b a h', a=num_agents)

        agent_type_embedings = self.agent_type__learnable_embeddings(agent_types) # (batch_size, num_agents, hidden_dim)
        agent_embeddings = agent_embeddings + agent_type_embedings

        return agent_embeddings



class MotionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_prob=0.1, num_motion_mode=6, num_time_steps=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_motion_mode = num_motion_mode
        self.num_time_steps = num_time_steps
        self.learnale_motion_query = nn.Parameter(torch.zeros(num_motion_mode, hidden_dim)) # (num_motion_mode, hidden_dim) (K, D)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
        self.trajs_projection = nn.Linear(in_features=hidden_dim, out_features=2 * self.num_time_steps)
        self.score_projection = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, agent_embeddings):
        batch_size, num_agents, _ = agent_embeddings.size()
        query = self.learnale_motion_query
        query = einops.repeat(query, 'k d -> (b n) k d', b=batch_size, n=num_agents)
        agent_embeddings = einops.rearrange(agent_embeddings, 'b n d -> (b n) d', b=batch_size, n=num_agents)
        agent_embeddings = einops.repeat(agent_embeddings, 'n d -> n k d', k=self.num_motion_mode)
        key = agent_embeddings
        value = agent_embeddings
        # mask the agent with other batch
        mask = torch.zeros(batch_size, num_agents, self.num_motion_mode, num_agents)
        context, _ = self.cross_attention.forward(query=query, key=key, value=value)
        context = context.view(batch_size, num_agents, self.num_motion_mode, self.hidden_dim)
        trajs = self.trajs_projection(context)
        scores = self.score_projection(context)
        trajs = trajs.view(batch_size, num_agents,  self.num_motion_mode, self.num_time_steps, 2)
        scores = scores.view(batch_size, num_agents, self.num_motion_mode)
        return trajs, scores, context
    

class MotionPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_states = config.num_states
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout_prob = config.dropout_prob
        self.num_layers = config.num_layers # number of layers in the transformer encoder
        self.num_motion_mode = config.num_motion_mode
        self.num_time_steps = config.num_time_steps
        self.encoder = MotionEncoder(num_states=self.num_states, hidden_dim=self.hidden_dim, num_heads=self.num_heads, dropout_prob=self.dropout_prob, num_layers=self.num_layers)
        self.decoder = MotionDecoder(hidden_dim=self.hidden_dim, num_heads=self.num_heads, dropout_prob=self.dropout_prob, num_motion_mode=self.num_motion_mode, num_time_steps=self.num_time_steps)
    
    def forward(self, agent_states, agent_types, valid_mask):
        """
        agent_states: torch.tesor (batch_size, num_agents, num_timesteps, num_states) 
        agent_types: torch.tensor (batch_size, num_agents)
        valid_mask: torch.tensor (batch_size, num_agents, num_timesteps)
        """

        agent_embeddings = self.encoder(agent_states, agent_types, valid_mask) # (batch_size, num_agents, hidden_dim)
        trajs, scores, context = self.decoder(agent_embeddings)
        
        return trajs, scores, context, agent_embeddings
    

if __name__ == "__main__":
    pass
