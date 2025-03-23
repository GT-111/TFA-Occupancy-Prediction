import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    """
    A single ConvGRUCell processes one timestep's input:
      x      => (B, C_in,  H, W)
      h_prev => (B, C_hid, H, W)
    and outputs one updated hidden state:
      h_new  => (B, C_hid, H, W)
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        padding = kernel_size // 2
        # Combined conv for update gate (z) and reset gate (r)
        self.conv_zr = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2 * hidden_channels,  # for z, r
            kernel_size=kernel_size,
            padding=padding
        )
        # Conv for candidate hidden state
        self.conv_h_tilde = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev=None):
        """
        x:      (B, C_in,  H, W)
        h_prev: (B, C_hid, H, W) or None
        returns:
          h_new: (B, C_hid, H, W)
        """
        B, _, H, W = x.shape

        if h_prev is None:
            # Initialize hidden state with zeros
            h_prev = torch.zeros(B, self.hidden_channels, H, W, device=x.device)

        # 1. Concat input + prev hidden along channels
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_hid, H, W)

        # 2. Compute update (z) and reset (r) gates
        zr = self.conv_zr(combined)               # (B, 2*C_hid, H, W)
        z, r = torch.split(zr, self.hidden_channels, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # 3. Candidate hidden state
        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h_tilde(combined_r))

        # 4. Final hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new
    
    
class MemoryConvGRU(nn.Module):
    """
    Multi-layer ConvGRU that processes exactly ONE timestep per forward() call.
    Follows a similar style to the TimeSeriesMemoryGRU you provided.
    """
    def __init__(self, config):
        """
        config should have:
          config.prev_out_channels: int => channels in prev_output
          config.context_channels:  int => channels in context
          config.hidden_channels:   list[int] or int => hidden dim(s) for each layer
          config.num_layers:        int => how many layers
          config.kernel_size:       int => convolution kernel size
        """
        super().__init__()
        # self.prev_out_channels = config.prev_out_channels
        self.context_channels  = config.context_channels

        # You can allow config.hidden_channels to be either a single int
        # or a list of length config.num_layers
        if isinstance(config.hidden_channels, int):
            # same hidden dim for all layers
            self.hidden_channels = [config.hidden_channels]*config.num_layers
        else:
            self.hidden_channels = config.hidden_channels
        
        self.num_layers   = config.num_layers
        self.kernel_size  = config.kernel_size

        # Build a stack of ConvGRUCell(s)
        self.cells = nn.ModuleList()
        
        # First layer sees channels = prev_out_channels + context_channels
        # input_channels = self.prev_out_channels + self.context_channels
        input_channels = self.context_channels
        self.cells.append(ConvGRUCell(input_channels,
                                      self.hidden_channels[0],
                                      self.kernel_size))
        
        # Each subsequent layer sees the hidden output of the previous layer
        for i in range(1, self.num_layers):
            in_ch = self.hidden_channels[i-1]
            out_ch = self.hidden_channels[i]
            self.cells.append(ConvGRUCell(in_ch, out_ch, self.kernel_size))
    def init_hidden(self, batch_size, H, W, device):
        """
        Initialize hidden states for all layers
        """
        hidden_state = []
        for i in range(self.num_layers):
            hidden_state.append(torch.zeros(batch_size, self.hidden_channels[i], H, W, device=device))
        return hidden_state
    
    def forward(self, context, hidden_state):
        """
        Args:
          prev_output: (B, C_prev, H, W)
            The model’s output from the previous timestep (or ground-truth).
          context:     (B, C_ctx,  H, W)
            Covariates / context for the current timestep.
          hidden_state: (num_layers, B, C_hid, H, W)
            The previous hidden states for each layer.

        Returns:
          updated_memory: (B, C_hid_top, H, W)
            The top layer’s new hidden output for the current timestep.
          updated_hidden: (num_layers, B, C_hid, H, W)
            Updated hidden states for all layers.
        """
        # 1) Prepare input for layer-0 by concatenating prev_output & context
        #    shape => (B, C_prev + C_ctx, H, W)
        # x = torch.cat([prev_output, context], dim=1)
        x = context

        # hidden_state is shape: (num_layers, B, C_hid, H, W)
        # We'll store updated hidden states in a list
        new_hidden_list = []

        # 2) Forward pass for each layer
        current_input = x
        for layer_idx, cell in enumerate(self.cells):
            # hidden for this layer => (B, C_hid_layer, H, W)
            h_prev = hidden_state[layer_idx] if hidden_state is not None else None

            # 2a) Pass current_input + h_prev through the cell
            h_new = cell(current_input, h_prev)
            
            # 2b) The new hidden becomes the input for the next layer
            current_input = h_new

            # keep track of new hidden state
            new_hidden_list.append(h_new)
        
        # 3) The top layer’s output => updated_memory
        updated_memory = new_hidden_list[-1]
        
        # 4) Stack new hidden states into shape (num_layers, B, C_hid, H, W)
        updated_hidden = new_hidden_list
        
        return updated_memory, updated_hidden
