import torch.nn as nn

class Task():

    def __init__(self, model:nn.Module, device):
        self.model = model
        self.device = device
    
    def preprocess(self, input_data_dic):
        raise NotImplementedError("The method must be implemented")
    
    def postprocess(self, prediction_data_dic):
        raise NotImplementedError("The method must be implemented")
    


    def forward_one_step(self, input_data_dic):

        history_data_dic, ground_truth_data_dic = self.preprocess(input_data_dic)
        prediction_data_dic = self.model(history_data_dic)
        prediction_data_dic = self.postprocess(prediction_data_dic)

        return ground_truth_data_dic, prediction_data_dic
    

        
    