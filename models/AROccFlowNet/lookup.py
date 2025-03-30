from json import decoder
import torch
import torch.nn as nn
import einops

class MemoryModule(nn.Module):
    """
    Stores a buffer of up to 'k' previous feature maps, 
    and uses cross-attention to fuse the current feature map 
    with the stored memory.
    """
    def __init__(self, d_model=256, n_heads=4, n_layers=1, memory_size=4):
        """
        Args:
          d_model: size of token embeddings for the cross-attention
          n_heads: number of attention heads
          n_layers: number of transformer encoder layers
          memory_size: k (maximum number of past steps to store)
        """
        super().__init__()
        self.memory_size = memory_size  # "k"

        # We'll store past features in a simple list (Python queue).
        # Each element is (B, C, H, W) before flattening.
        self.memory_buffer = []



        # A single (or stack of) TransformerEncoderLayer for cross-attn
        # Because "TransformerEncoder" actually does self-attention, we do a trick:
        # We'll treat memory + current as a single "batch of tokens" and
        # use different attention masks or grouping if needed. 
        # Another approach is to build a small "cross-attention" block manually.

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=2*d_model, 
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=n_layers
        )

    def forward(self, current_fused_features):
        """
        Args:
          current_fused_features: (B, C, H, W) for the *current* timestep

        Returns:
          updated_features: (B, C, H, W) after cross-attending with memory
        """
        B, C, H, W = current_fused_features.shape
        device = current_fused_features.device

        # ------ 1) Convert current features to embedding tokens ------
        # Project to d_model if needed
        # (B, C, H, W) -> (B, d_model, H, W)
        # Flatten spatial dimensions => tokens: (B, H*W, d_model)
        x_cur_tokens = einops.rearrange(current_fused_features, 'b d h w -> b (h w) d')


        # Now we combine them all into a single token sequence
        # e.g., [ (B, L, d_model), (B, L, d_model), ... ]
        # where L = H*W
        if len(self.memory_buffer) > 0:
            mem_tokens = torch.cat(self.memory_buffer, dim=1)  # (B, k*L, d_model)
            # We'll do self-attention over [memory + current]
            all_tokens = torch.cat([mem_tokens, x_cur_tokens], dim=1)  # (B, M + L, d_model)
        else:
            # No memory yet, just use current tokens
            all_tokens = x_cur_tokens  # (B, L, d_model)

        # ------ 3) Apply Transformer (self-attn) ------
        # A simple approach is to let the entire sequence attend to itself.
        # This is effectively letting "current tokens" cross-attend to "memory tokens" 
        # and also letting memory tokens attend among themselves. 
        # For stricter cross-attn, you'd build a custom mask or a cross-attn module.
        # all_tokens_encoded = self.transformer_encoder(all_tokens)  # (B, M+L, d_model) or (B, L, d_model)
        current_tokens_encoded = self.transformer_decoder.forward(tgt=x_cur_tokens, memory=all_tokens)
      
        # Reshape back to (B, d_model, H, W)
        updated_features = einops.rearrange(
            current_tokens_encoded, 
            'b (h w) d -> b d h w', 
            h=H, w=W
        )

        # ------ 4) Update memory buffer ------
        # Append the *raw fused_features* (before transform) or the newly updated features?
        # Commonly you'd store the *original* fused_features, so let's do that:
        self.memory_buffer.append(x_cur_tokens.detach())  # no grad
        if len(self.memory_buffer) > self.memory_size:
            self.memory_buffer.pop(0)  # drop the oldest

        return updated_features
