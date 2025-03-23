import torch

class SampleModelInput():

    def __init__(self, dataset_config):
        self.batch_size = 3
        self.num_agents = 20
        self.num_states = dataset_config.trajectory.num_states
        self.num_his_points = dataset_config.task.num_his_points
        self.num_waypoints = dataset_config.task.num_waypoints
        self.occupancy_flow_map_config = dataset_config.occupancy_flow_map
        self.occupancy_flow_map_height = self.occupancy_flow_map_config.occupancy_flow_map_height
        self.occupancy_flow_map_width = self.occupancy_flow_map_config.occupancy_flow_map_width
        self.num_agent_type = 6
        # vector_features_list = ['length', 'width', 'class', 'direction']
        # node_features_list = ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle']
    def generate_agent_states_and_types(self):
        agent_states = torch.rand(self.batch_size, self.num_agents, self.num_his_points, self.num_states)
        agent_types = torch.randint(0, self.num_agent_type, (self.batch_size, self.num_agents))
        return agent_states, agent_types
    
    def generate_occupancy_flow_maps(self, num_steps):
        # occupancy_map: Occupancy map input (B, H, W, T, C)
        # flow_map: Flow map input (B, H, W, T, C)
        occupancy_map = torch.rand(self.batch_size, self.occupancy_flow_map_height, self.occupancy_flow_map_width, num_steps, 1)
        flow_map = torch.rand(self.batch_size,self.occupancy_flow_map_height, self.occupancy_flow_map_width, num_steps -1 , 2)

        return occupancy_map, flow_map

    def generate_sample_input(self, device):
        result_dic = {}
        agent_states, agent_types = self.generate_agent_states_and_types()
        his_occupancy_map, his_flow_map = self.generate_occupancy_flow_maps(self.num_his_points)
        pred_occupancy_map, pred_flow_map = self.generate_occupancy_flow_maps(self.num_waypoints)
        result_dic['cur/his/agent_states'] = agent_states.to(device)
        result_dic['cur/his/agent_types'] = agent_types.to(device)
        result_dic['cur/his/valid_mask'] = torch.ones(self.batch_size, self.num_agents, self.num_his_points).to(device)
        result_dic['cur/his/occupancy_map'] = his_occupancy_map.to(device)
        result_dic['cur/his/flow_map'] = his_flow_map.to(device)
        result_dic['nxt/his/occupancy_map'] = his_occupancy_map.to(device)
        result_dic['nxt/his/flow_map'] = his_flow_map.to(device)
        result_dic['prv/his/occupancy_map'] = his_occupancy_map.to(device)
        result_dic['prv/his/flow_map'] = his_flow_map.to(device)

        result_dic['cur/pred/occupancy_map'] = pred_occupancy_map.to(device)
        result_dic['cur/pred/flow_map'] = pred_flow_map.to(device)
        result_dic['nxt/pred/occupancy_map'] = pred_occupancy_map.to(device)
        result_dic['nxt/pred/flow_map'] = pred_flow_map.to(device)
        result_dic['prv/pred/occupancy_map'] = pred_occupancy_map.to(device)
        result_dic['prv/pred/flow_map'] = pred_flow_map.to(device)
        # print the setting information
        print('==================== Sample Input Information ====================')
        print('Batch Size:', self.batch_size)
        print('Number of Agents:', self.num_agents)
        print('Number of Timesteps:', self.num_his_points)
        print('Number of States:', self.num_states)
        print('Image Size:', (self.occupancy_flow_map_height, self.occupancy_flow_map_width))
        print('Number of Agent Types:', self.num_agent_type)
        print('Agent States Shape:', agent_states.shape)
        print('Agent Types Shape:', agent_types.shape)
        print('=================================================================')
        return result_dic
