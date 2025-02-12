# Custom Imlpementation of the Effcient Muotion Prediction (EMP) module
# Since the I24 has no road map, the lane encoder is not used in this implementation
# For details, please refer to the original paper: https://arxiv.org/abs/2409.16154

from random import sample
from turtle import forward
import torch
import torch.nn as nn
import einops


class EfficientMotionPredictor(nn.Module):

    def __init__(self, num_states, hidden_dim, num_heads, dropout_prob=0.1, num_layers=1, num_motion_mode=6):
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
        self.temporal_attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dropout=self.dropout_prob), num_layers=self.num_layers)
        self.agent_type__learnable_embeddings = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=hidden_dim)
        self.agent_type__learnable_embeddings.weight.data = nn.init.xavier_uniform_(self.agent_type__learnable_embeddings.weight.data)
        
        # TODO: Add the positional encoding

        
    def forward(self, agent_states, agent_types):
        """
        agent_states: torch.tesor (batch_size, num_agents, num_timesteps, num_states) 
        agent_types: torch.tensor (batch_size, num_agents)
        """
        batch_size, num_agents, num_timesteps, num_states = agent_states.size()
        agent_embeddings = self.agent_projection(agent_states) # (batch_size, num_agents, num_timesteps, hidden_dim)
        agent_embeddings = einops.rearrange(agent_embeddings, 'b a t h -> (b a) t h')

        agent_embeddings = self.temporal_attention.forward(agent_embeddings)
        agent_embeddings = einops.reduce(agent_embeddings, 'b t h -> b h', 'max')
        agent_embeddings = einops.rearrange(agent_embeddings, '(b a) h -> b a h', a=num_agents)

        agent_type_embedings = self.agent_type__learnable_embeddings(agent_types) # (batch_size, num_agents, hidden_dim)
        agent_embeddings = agent_embeddings + agent_type_embedings

        return agent_embeddings


# TODO: Test the MotionDecoder
class MotionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_prob=0.1, num_motion_mode=6):
        super().__init__()
        self.learnale_motion_query = nn.Parameter(torch.zeros(num_motion_mode, hidden_dim)) # (num_motion_mode, hidden_dim) (K, D)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_prob)
        self.trajs_projection = nn.Linear(in_features=hidden_dim, out_features=2)
        self.score_projection = nn.Linear(in_features=hidden_dim, out_features=1)
    def forward(self, agent_embeddings):
        query = self.learnale_motion_query
        key = agent_embeddings
        value = agent_embeddings
        context, _ = self.cross_attention(query=query, key=key, value=value)
        trajs = self.trajs_projection(context)
        scores = self.score_projection(context)

        return trajs, scores, context
from utils.SampleModelInput import SampleModelInput
if __name__ == "__main__":

    agent_states = SampleModelInput().generate_sample_input()
    batch_size, num_agents, num_timesteps, num_states = agent_states.size()
    emp = EfficientMotionPredictor(num_states=num_states, hidden_dim=64, num_heads=4)

    print(f'Input shape: {agent_states.size()}')
    emp(agent_states)