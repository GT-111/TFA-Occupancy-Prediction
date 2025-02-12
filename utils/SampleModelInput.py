import torch

class SampleModelInput():
    batch_size: int
    num_agents: int
    num_timesteps: int
    num_states: int

    def __init__(self):
        self.batch_size = 8
        self.num_agents = 20
        self.num_timesteps = 10
        self.num_states = 10
        # vector_features_list = ['length', 'width', 'class', 'direction']
        # node_features_list = ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle']
    def generate_sample_input(self):
        return torch.rand(self.batch_size, self.num_agents, self.num_timesteps, self.num_states)