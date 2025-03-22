import torch
import math

def generate_1d_sin_pos_embedding(T, D, device):
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

    return pe.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)  # Shape (1, 1, 1, T, D)

def generate_2d_sin_pos_embedding(H, W, d_model, device):
    """
    Generate a 2D sinusoidal positional embedding of shape [H, W, d_model].
    
    - The first `d_model//2` channels encode the row position (h).
    - The last `d_model//2` channels encode the column position (w).
    
    Args:
        H (int): Height of the grid.
        W (int): Width of the grid.
        d_model (int): Total embedding dimension.

    Returns:
        pos_embedding (torch.Tensor): Shape [H, W, d_model]
    """
    assert d_model % 2 == 0, "d_model must be even"

    # Split embedding dimension between row and column encodings
    d_model_row = d_model // 2
    d_model_col = d_model - d_model_row

    # Create coordinate tensors for row and column indices
    rows = torch.arange(H).float().unsqueeze(1)  # Shape: [H, 1]
    cols = torch.arange(W).float().unsqueeze(1)  # Shape: [W, 1]

    # Compute sinusoidal frequencies (div_term)
    div_term_row = torch.exp(torch.arange(0, d_model_row, 2).float() * (-math.log(10000.0) / d_model_row))
    div_term_col = torch.exp(torch.arange(0, d_model_col, 2).float() * (-math.log(10000.0) / d_model_col))

    # Compute sinusoidal embeddings for rows
    row_pe = torch.zeros(H, d_model_row)
    row_pe[:, 0::2] = torch.sin(rows * div_term_row)  # Apply sin to even indices
    row_pe[:, 1::2] = torch.cos(rows * div_term_row)  # Apply cos to odd indices

    # Compute sinusoidal embeddings for columns
    col_pe = torch.zeros(W, d_model_col)
    col_pe[:, 0::2] = torch.sin(cols * div_term_col)
    col_pe[:, 1::2] = torch.cos(cols * div_term_col)

    # Expand row_pe and col_pe to match the grid shape [H, W, d_model]
    row_pe_broadcast = row_pe.unsqueeze(1).expand(H, W, d_model_row)  # [H, W, d_model_row]
    col_pe_broadcast = col_pe.unsqueeze(0).expand(H, W, d_model_col)  # [H, W, d_model_col]

    # Concatenate along the last dimension
    pos_embedding = torch.cat([row_pe_broadcast, col_pe_broadcast], dim=-1)  # [H, W, d_model]

    return pos_embedding.to(device)



def generate_3d_sinusoidal_embeddings(T, H, W, d, device):
    """
    Generate 3D sinusoidal positional embeddings for a (T, H, W) volume.

    The returned embedding has shape (T, H, W, d), where d is distributed among T, H, and W.

    Args:
        T (int): Temporal (or depth) dimension.
        H (int): Height dimension.
        W (int): Width dimension.
        d (int): Total dimensionality of the embedding.
        device (str or torch.device): The device on which to place the resulting tensor.

    Returns:
        torch.Tensor: A tensor of shape (T, H, W, d)
    """
    # Split d into three parts, giving priority to T, H, and W sequentially
    d_t = d // 3
    d_h = d // 3
    d_w = d - (d_t + d_h)  # Ensure the sum matches d
    
    # 1. Create coordinate grids for T, H, W: each shaped (T, H, W)
    grid_t = torch.arange(T, device=device).view(T, 1, 1).expand(T, H, W)  # [0..T-1]
    grid_h = torch.arange(H, device=device).view(1, H, 1).expand(T, H, W)  # [0..H-1]
    grid_w = torch.arange(W, device=device).view(1, 1, W).expand(T, H, W)  # [0..W-1]

    def sinusoidal_encoding(coords, d_model):
        """
        Build standard sinusoidal encoding for one axis (t, h, or w).

        Args:
            coords (torch.Tensor): shape (T, H, W), integer positions along that axis.
            d_model (int): number of channels to devote to this axis.

        Returns:
            Tensor of shape (T, H, W, d_model).
        """
        if d_model == 0:
            return None  # Skip if no dimensions are assigned

        # Expand coords to (T, H, W, 1)
        coords = coords.unsqueeze(-1)  # shape (T, H, W, 1)
        
        # Create channel indices [0..d_model-1], shape (d_model,)
        channel_idx = torch.arange(d_model, device=device).unsqueeze(0)  # (1, d_model)

        # Compute scaling factor:
        div_term = torch.pow(10000.0, -channel_idx / max(1, d_model))  # (1, d_model)

        # Broadcast to (T, H, W, d_model)
        angles = coords * div_term  # shape (T, H, W, d_model)

        # Even indices -> sin, odd indices -> cos
        sin_mask = (channel_idx % 2 == 0)  # shape (1, d_model)
        sin_mask_expanded = sin_mask.expand_as(angles)

        encoding = torch.zeros_like(angles)
        encoding[sin_mask_expanded] = angles[sin_mask_expanded].sin()
        encoding[~sin_mask_expanded] = angles[~sin_mask_expanded].cos()

        return encoding

    # 2. Compute sinusoidal embeddings for t, h, and w (skip any axis with d=0)
    t_encoding = sinusoidal_encoding(grid_t, d_t) if d_t > 0 else None
    h_encoding = sinusoidal_encoding(grid_h, d_h) if d_h > 0 else None
    w_encoding = sinusoidal_encoding(grid_w, d_w) if d_w > 0 else None

    # 3. Concatenate along the last dimension -> (T, H, W, d)
    encodings = [e for e in [t_encoding, h_encoding, w_encoding] if e is not None]
    embedding_3d = torch.cat(encodings, dim=-1) if len(encodings) > 1 else encodings[0]

    return embedding_3d