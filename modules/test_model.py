import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        # Initial layers for occ
        self.occ_conv = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Initial layers for flow
        self.flow_conv = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1),  # Expecting flow to be reshaped to 4D
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        
        # Shared convolution layers after combining features
        self.shared_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Output layers for occupancy and flow
        self.observed_occ_out = nn.Conv2d(32, 8, kernel_size=1)
        self.occluded_occ_out = nn.Conv2d(32, 8, kernel_size=1)
        self.flow_out = nn.Conv2d(32, 16, kernel_size=1)  # We will reshape this back to [10, 8, 256, 128, 2] later
        
    
    def forward(self, occ, flow):
        # Process occupancy (occ)
        occ_features = self.occ_conv(occ)  # Shape: [10, 16, 256, 128]
        
        # Reshape flow to merge last dimension, then process with conv layers
        flow = flow.view(flow.size(0), -1, flow.size(2), flow.size(3))  # Shape: [10, 80, 256, 128]
        flow_features = self.flow_conv(flow)  # Shape: [10, 16, 256, 128]
        
        # Concatenate occ and flow features
        combined_features = torch.cat((occ_features, flow_features), dim=1)  # Shape: [10, 32, 256, 128]
        
        # Shared convolutions
        shared_features = self.shared_conv(combined_features)  # Shape: [10, 32, 256, 128]
        
        # Separate outputs for occ and flow
        observed_occ_out = self.observed_occ_out(shared_features)  # Shape: [10, 8, 256, 128]
        occluded_occ_out = self.occluded_occ_out(shared_features)
        flow_out = self.flow_out(shared_features)  # Shape: [10, 16, 256, 128]
        flow_out = flow_out.view(flow_out.size(0), 8, 256, 128, 2)  # Reshape to [10, 8, 256, 128, 2]
        
        return observed_occ_out, occluded_occ_out, flow_out

