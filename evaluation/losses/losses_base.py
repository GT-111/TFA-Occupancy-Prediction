class Loss():
    def __init__(self):
        
        raise NotImplementedError("Initialization method not implemented. Please implement it in the derived class.")
    
    def compute(self, prediction_dict, ground_truth_dict):
        
        raise NotImplementedError("Compute method not implemented. Please implement it in the derived class.")
    
        return loss_dict
    
    def update(self, loss_dict):
        
        raise NotImplementedError("Update method not implemented. Please implement it in the derived class.")
    
    def get_result(self):
        
        raise NotImplementedError("Reset method not implemented. Please implement it in the derived class.")

        return result_dic
    def reset(self):
        raise NotImplementedError("Reset method not implemented. Please implement it in the derived class.")