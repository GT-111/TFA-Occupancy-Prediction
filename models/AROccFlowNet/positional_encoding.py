import torch
import math

def positional_encoding(T, D):
    """
    Generate sinusoidal positional encodings for T timesteps and D dimensions.
    Args:
        T: Number of time steps
        D: Coordinate dimension (e.g., 2 for XY)
    Returns:
        pos_encoding: (1, 1, 1, T, D) positional encoding for broadcasting
    """
    position = torch.arange(T, dtype=torch.float).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float) * (-math.log(10000.0) / D))
    
    pe = torch.zeros(T, D)  # (T, D)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 1, T, D)
