import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AROccFlowNet.PositionalEncoding import positional_encoding
from utils.config import load_config
# Q^A [8, 20, 64]
# Q_M^A [8, 20, 6, 64] 
# H_M^A[8, 20, 6, 10, 2]
# S_M^A[8, 20, 6]
# q_a = torch.rand(8, 20, 64)
# q_m_a = torch.rand(8, 20, 6, 64)
# s_m_a = torch.rand(8, 20, 6)

# pos_encoding = positional_encoding(T=10, D=2)  # Shape (1, 1, 1, 10, 2)

# predicted_trajs = torch.randn(8, 20, 6, 10, 2)  # Example input
# predicted_trajs_with_pe = predicted_trajs + pos_encoding  # Broadcasting addition
# linear1 = nn.Linear(2, 64)
# h_m_a = einops.reduce(linear1(predicted_trajs_with_pe), 'b a m t h -> b a m h', 'max')
# # claculate the weighted sum use score
# score = einops.rearrange(F.softmax(s_m_a, dim=-1), 'b a m -> b a m 1')

# weighted_sum = torch.matmul(einops.rearrange((h_m_a+q_m_a), 'b a m h -> b a h m'), score)
# weighted_sum = einops.rearrange(weighted_sum, 'b a d 1 -> b a d')
# linear2 = nn.Linear(64, 64)
# print(weighted_sum.shape)
# h_a_traj = linear2(weighted_sum + q_a)
# print(h_a_traj.shape) 
# class AROccFlowNet(nn.Module):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
config = load_config("configs/AROccFlowNetS.py")
print(config)