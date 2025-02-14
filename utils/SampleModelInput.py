import torch

class SampleModelInput():
    batch_size: int
    num_agents: int
    num_timesteps: int
    num_states: int

    def __init__(self):
        self.batch_size = 4
        self.num_agents = 20
        self.num_timesteps = 10
        self.num_states = 10
        self.img_size = (256, 256)
        self.num_agent_type = 6
        # vector_features_list = ['length', 'width', 'class', 'direction']
        # node_features_list = ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle']
    def generate_agent_states_and_types(self):
        agent_states = torch.rand(self.batch_size, self.num_agents, self.num_timesteps, self.num_states)
        agent_types = torch.randint(0, self.num_agent_type, (self.batch_size, self.num_agents))
        return agent_states, agent_types
    
    def generate_occupancy_flow_maps(self):
        occupancy_map = torch.rand(self.batch_size, self.num_timesteps, 1, self.img_size[0], self.img_size[1])
        flow_map = torch.rand(self.batch_size, self.num_timesteps, 2, self.img_size[0], self.img_size[1])

        return occupancy_map, flow_map

    def generate_sample_input(self):
        result_dic = {}
        agent_states, agent_types = self.generate_agent_states_and_types()
        occupancy_map, flow_map = self.generate_occupancy_flow_maps()
        result_dic['cur/his/agent_states'] = agent_states
        result_dic['cur/his/agent_types'] = agent_types
        result_dic['cur/his/occupancy_map'] = occupancy_map
        result_dic['cur/his/flow_map'] = flow_map
        result_dic['nxt/his/occupancy_map'] = occupancy_map
        result_dic['nxt/his/flow_map'] = flow_map
        result_dic['prv/his/occupancy_map'] = occupancy_map
        result_dic['prv/his/flow_map'] = flow_map
        # print the setting information
        print('==================== Sample Input Information ====================')
        print('Batch Size:', self.batch_size)
        print('Number of Agents:', self.num_agents)
        print('Number of Timesteps:', self.num_timesteps)
        print('Number of States:', self.num_states)
        print('Image Size:', self.img_size)
        print('Number of Agent Types:', self.num_agent_type)
        print('Agent States Shape:', agent_states.shape)
        print('Agent Types Shape:', agent_types.shape)
        print('Occupancy Map Shape:', occupancy_map.shape)
        print('Flow Map Shape:', flow_map.shape)
        print('=================================================================')
        return result_dic
