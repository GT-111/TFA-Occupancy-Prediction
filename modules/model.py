import numpy as np
import typing
from utils.file_utils import get_config

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import enum

config = get_config()
class ResamplingType(enum.Enum):
  NEAREST = 0
  BILINEAR = 1

class BorderType(enum.Enum):
  ZERO = 0
  DUPLICATE = 1

class PixelType(enum.Enum):
  INTEGER = 0
  HALF_INTEGER = 1

import enum
from typing import Optional
from typing import Union, Sequence

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, torch.Tensor]

def sample(image: torch.Tensor,
           warp: torch.Tensor,
           resampling_type: ResamplingType = ResamplingType.BILINEAR,
           border_type: BorderType = BorderType.ZERO,
           pixel_type: PixelType = PixelType.HALF_INTEGER) -> torch.Tensor:
    """Samples an image at user defined coordinates.

    Note:
        The warp maps target to source. In the following, A1 to An are optional
        batch dimensions.

    Args:
        image: A tensor of shape `[B, H_i, W_i, C]`, where `B` is the batch size,
        `H_i` the height of the image, `W_i` the width of the image, and `C` the
        number of channels of the image.
        warp: A tensor of shape `[B, A_1, ..., A_n, 2]` containing the x and y
        coordinates at which sampling will be performed. The last dimension must
        be 2, representing the (x, y) coordinate where x is the index for width
        and y is the index for height.
    resampling_type: Resampling mode. Supported values are
        `ResamplingType.NEAREST` and `ResamplingType.BILINEAR`.
        border_type: Border mode. Supported values are `BorderType.ZERO` and
        `BorderType.DUPLICATE`.
        pixel_type: Pixel mode. Supported values are `PixelType.INTEGER` and
        `PixelType.HALF_INTEGER`.
        name: A name for this op. Defaults to "sample".

    Returns:
        Tensor of sampled values from `image`. The output tensor shape
        is `[B, A_1, ..., A_n, C]`.

    Raises:
        ValueError: If `image` has rank != 4. If `warp` has rank < 2 or its last
        dimension is not 2. If `image` and `warp` batch dimension does not match.
    """

    # shape.check_static(image, tensor_name="image", has_rank=4)
    # shape.check_static(
    #     warp,
    #     tensor_name="warp",
    #     has_rank_greater_than=1,
    #     has_dim_equals=(-1, 2))
    # shape.compare_batch_dimensions(
    #     tensors=(image, warp), last_axes=0, broadcast_compatible=False)

    if pixel_type == PixelType.HALF_INTEGER:
        warp -= 0.5

    if resampling_type == ResamplingType.NEAREST:
        warp = torch.math.round(warp)

    if border_type == BorderType.ZERO:
        image = F.pad(image, (0,0,1,1,1,1,0,0))
        warp = warp + 1

    warp_shape = warp.size()
    flat_warp = torch.reshape(warp, (warp_shape[0], -1, 2))
    flat_sampled = _interpolate_bilinear(image, flat_warp, indexing="xy")
    output_shape = [*warp_shape[:-1], flat_sampled.size()[-1]]
    return torch.reshape(flat_sampled, output_shape)

def _interpolate_bilinear(
    grid,
    query_points,
    indexing,
):
    """pytorch implementation of tensorflow interpolate_bilinear."""
    device = grid.device
    grid_shape = grid.size()
    query_shape = query_points.size()

    batch_size, height, width, channels = (
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        grid_shape[3],
    )

    num_queries = query_shape[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = torch.unbind(query_points, dim=2)

    for i, dim in enumerate(index_order):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = grid_shape[i + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type, device=device)
        min_floor = torch.tensor(0.0, dtype=query_type, device=device)
        floor = torch.minimum(
            torch.maximum(min_floor, torch.floor(queries)), max_floor
        )
        int_floor = floor.to(torch.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).to(grid_type)
        min_alpha = torch.tensor(0.0, dtype=grid_type, device=device)
        max_alpha = torch.tensor(1.0, dtype=grid_type, device=device)
        alpha = torch.minimum(torch.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)

        flattened_grid = torch.reshape(grid, [batch_size * height * width, channels])
        batch_offsets = torch.reshape(
            torch.arange(0, batch_size, device=device) * height * width, [batch_size, 1]
        )

    # This wraps tf.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using tf.gather_nd.
    def gather(y_coords, x_coords, name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = flattened_grid[linear_coordinates]
        return torch.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    # now, do the actual interpolation
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp

class MultiHeadAttention(nn.Module):
    r"""MultiHead Attention layer.
    Defines the MultiHead Attention operation as described in
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes
    in the tensors `query`, `key`, and `value`, and returns the dot-product attention
    between them:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 4) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 5) # (batch_size, key_elements, key_depth)
    >>> value = np.random.rand(3, 6, 6) # (batch_size, key_elements, value_depth)
    >>> attention = mha([query, key, value]) # (batch_size, query_elements, value_depth)
    >>> attention.shape
    TensorShape([3, 5, 6])
    If `value` is not given then internally `value = key` will be used:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 5) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 10) # (batch_size, key_elements, key_depth)
    >>> attention = mha([query, key]) # (batch_size, query_elements, key_depth)
    >>> attention.shape
    TensorShape([3, 5, 10])
    Args:
        head_size: int, dimensionality of the `query`, `key` and `value` tensors
            after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the
            input dimension of `value` or `key` will be used,
            default `None`.
        dropout: float, `rate` parameter for the dropout layer that is
            applied to attention after softmax,
        default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear
            output projection.
        return_attn_coef: bool, if `True`, return the attention coefficients as
            an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.
    Call Args:
        inputs:  List of `[query, key, value]` where
            * `query`: Tensor of shape `(..., query_elements, query_depth)`
            * `key`: `Tensor of shape '(..., key_elements, key_depth)`
            * `value`: Tensor of shape `(..., key_elements, value_depth)`, optional, if not given `key` will be used.
        mask: a binary Tensor of shape `[batch_size?, num_heads?, query_elements, key_elements]`
        which specifies which query elements can attendo to which key elements,
        `1` indicates attention and `0` indicates no attention.
    Output shape:
        * `(..., query_elements, output_size)` if `output_size` is given, else
        * `(..., query_elements, value_depth)` if `value` is given, else
        * `(..., query_elements, key_depth)`
    """

    def __init__(
        self,
        input_channels,
        head_size: int,
        num_heads: int,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
    ):
        super(MultiHeadAttention, self).__init__()

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = nn.Dropout(dropout)
        self._droput_rate = dropout

        num_query_features = input_channels[0]
        num_key_features = input_channels[1]
        num_value_features = (
            input_channels[2] if len(input_channels) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        self.query_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, num_query_features, self.head_size])))
        self.key_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, num_key_features, self.head_size])))
        self.value_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, num_key_features, self.head_size])))
        self.projection_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.zeros([self.num_heads, self.head_size, output_size])))


        if self.use_projection_bias:
            self.projection_bias = nn.Parameter(torch.zeros([output_size]))
        else:
            self.projection_bias = None

    def forward(self, inputs, mask=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        query = torch.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = torch.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = torch.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = torch.tensor(self.head_size, dtype=query.dtype).detach()
        query /= torch.sqrt(depth)

        # Calculate dot product attention
        logits = torch.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = mask.to(torch.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = torch.unsqueeze(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = F.softmax(logits, dim=-1)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef)

        # attention * value
        multihead_output = torch.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = torch.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

class FGMSA(nn.Module):
    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups=6,
        attn_drop=0., proj_drop=0., stride=1, 
        offset_range_factor=2, use_pe=True, dwc_pe=False,
        no_off=False, fixed_pe=False, stage_idx=3,use_last_ref=False,
        out_dim=384,fg=False,in_dim=384
    ):
        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.use_last_ref = use_last_ref
        self.fg = fg

        self.ref_res = None
        
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset_0 = nn.Conv2d(self.nc, self.nc, kernel_size=kk, stride=stride, padding="same", groups=self.n_groups)
        self.conv_norm = nn.LayerNorm(eps=1e-3, normalized_shape=self.nc)
        # self.out_norm = layers.LayerNormalization(1e-5)
        self.conv_offset_proj = nn.Conv2d(self.n_group_channels, 2, kernel_size=1,stride=1, bias=False)
        if self.fg:
            self.conv_offset_proj2 = nn.Conv2d(2, out_dim, kernel_size=1,stride=1)

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)

        self.proj_k = nn.Conv2d(self.nc, self.nc,kernel_size=1, stride=1)

        self.proj_v = nn.Conv2d(self.nc, self.nc,kernel_size=1, stride=1)

        self.proj_out = nn.Conv2d(self.nc, out_dim,kernel_size=1, stride=1)

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_pe:
            self.rpe_table = nn.Parameter(torch.fmod(torch.randn((self.kv_h * 2 - 1, self.kv_w * 2 - 1,self.n_heads)), 2) * 0.01, requires_grad=True)
        else:
            self.rpe_table = None
        
        dummy_x = torch.zeros((1,self.q_h,self.q_w,in_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim
        ref = torch.zeros((1*self.n_groups,self.q_h,self.q_w,2))
        self(dummy_x,last_reference=ref)
        summary(self)
    
    def _get_offset(self,x):
        # x [B, C, H, W]
        x = self.conv_offset_0(x) # [B, C, H, W]
        x = x.permute(0,2,3,1) # [B, H, W, C]
        x = torch.reshape(x,[-1, self.q_h*self.q_w, self.nc])
        x = self.conv_norm(x)
        x = torch.reshape(x,[-1, self.q_h,self.q_w, self.nc]) # [B, H, W, C]
        x = F.gelu(x)
        x = torch.reshape(torch.reshape(x,[-1,self.q_h,self.q_w,self.n_groups,self.n_group_channels]).permute([0,3,1,2,4]),[-1,self.q_h,self.q_w,self.n_group_channels])
        # x = torch.reshape(torch.reshape(x,[-1,self.q_h,self.q_w,self.n_groups,self.n_group_channels]).permute([0,3,1,2,4]),[-1,self.q_h,self.q_w,self.n_group_channels])
        x = x.permute(0,3,1,2) # channels as 2nd dim
        x = self.conv_offset_proj(x)
        x = x.permute(0,2,3,1) # channels as last dim
        return x
        
    
    def _get_ref_points(self, H_key, W_key, B):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H_key), 
            torch.arange(0, W_key),
            indexing="xy"
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref = ref.to(torch.float32)
        ref = torch.repeat_interleave(ref[np.newaxis,...],B*self.n_groups,dim=0)
        ref = ref.detach()
        return ref

    def forward(self, x,training=True,last_reference=None):
        B, H, W,C = x.size()
        x = x.permute(0,3,1,2) # channels as 2nd dim
        q = self.proj_q(x)
        offset = self._get_offset(q) # [B, H, W, 2]
        _,Hk,Wk,_ = offset.size()
        n_sample = Hk * Wk
        
        if self.offset_range_factor > 0:
            offset_range = torch.reshape(torch.tensor([Hk/2,Wk/2], device=x.device).detach(),(1, 1, 1, 2))
            offset = torch.tanh(offset)
            offset = torch.multiply(offset, offset_range)
            self.ref_res = torch.reshape(offset,(B,self.n_groups, Hk, Wk, 2))
        
        if self.fg:
            time_offset = torch.reshape(offset,(B * self.n_groups, Hk, Wk, 2))
            time_offset = time_offset.permute(0,3,1,2) # channels as 2nd dim
            flow_hidden = self.conv_offset_proj2(time_offset)
            flow_hidden = flow_hidden.permute(0,2,3,1) # channels last
            flow_hidden = torch.reshape(flow_hidden,(B, self.n_groups, Hk,Wk, self.out_dim))
        
        if self.use_last_ref:
            reference = torch.reshape(last_reference,(B*self.n_groups, Hk, Wk, 2))
        else:
            reference = self._get_ref_points(Hk, Wk, B).to(x.device).reshape((B*self.n_groups, Hk, Wk, 2))
            
        if self.no_off:
            offset = torch.zeros_like(offset, device=x.device)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = torch.tanh(offset + reference)
        
        x = x.permute(0,2,3,1) # channels last
        x = torch.reshape(torch.reshape(x,[B, H, W,self.n_groups,self.n_group_channels]).permute([0,3,1,2,4]),[B*self.n_groups, H, W,self.n_group_channels])
        
        warp = torch.concat([pos[...,1][...,np.newaxis],pos[...,0][...,np.newaxis]],dim=-1)
        x_sampled = sample(image=x, warp=warp,pixel_type=0)
        x_sampled = torch.reshape(torch.reshape(x, [B,self.n_groups, H, W,self.n_group_channels]).permute([0,2,3,1,4]),[B,n_sample,1,C])
        x_sampled = x_sampled.permute([0, 3, 1, 2]) # channels as 2nd dim
            
        q = torch.reshape(torch.reshape(q,(B, H * W,self.n_heads, self.n_head_channels)).permute([0,2,1,3]),[B*self.n_heads,H * W,self.n_head_channels])
        k = torch.reshape(torch.reshape(self.proj_k(x_sampled),(B, n_sample,self.n_heads, self.n_head_channels)).permute([0,2,1,3]),[B*self.n_heads,n_sample,self.n_head_channels])
        v = torch.reshape(torch.reshape(self.proj_v(x_sampled),(B, n_sample,self.n_heads, self.n_head_channels)).permute([0,2,1,3]),[B*self.n_heads,n_sample,self.n_head_channels])
        attn = torch.einsum('bqc, bkc-> bqk', q, k)
        attn = attn * self.scale
        
        if self.use_pe:
            rpe_table = self.rpe_table
            # rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
            rpe_bias = torch.repeat_interleave(rpe_table[np.newaxis,...],repeats=B,dim=0)
            
            q_grid = self._get_ref_points(H, W, B).to(x.device)
            
            displacement = torch.unsqueeze(torch.reshape(q_grid,(B * self.n_groups, H * W, 2)),dim=2) - torch.unsqueeze(torch.reshape(pos,(B * self.n_groups, n_sample, 2)),dim=1)

            rpe_bias = torch.reshape(rpe_bias, (B,2 * H - 1, 2 * W - 1,self.n_groups, self.n_group_heads)).permute([0,3,1,2,4])
            displacement = torch.concat([displacement[...,1][...,np.newaxis],displacement[...,0][...,np.newaxis]],dim=-1)
            
            attn_bias = sample(
                image=torch.reshape(rpe_bias,[B * self.n_groups,2 * H - 1, 2 * W - 1,self.n_group_heads]),
                warp=displacement,
                pixel_type=0
            )

            attn_bias = torch.reshape(attn_bias, [B*self.n_groups,H*W,n_sample,self.n_group_heads]).permute([0,3,1,2])
            
            attn_bias = torch.reshape(attn_bias,[B*self.n_heads,H*W,n_sample] )
            
            attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('bkv, bvc -> bck', attn, v)
        out = torch.reshape(out,(B,C,H,W))
        y = self.proj_drop(self.proj_out(out))
        y = y.permute([0,2,3,1]) # channels as last dim

        if self.fg:
            return y,torch.reshape(pos,(B, self.n_groups, Hk, Wk, 2)),flow_hidden
        
        return y, torch.reshape(pos,(B, self.n_groups, Hk, Wk, 2)), torch.reshape(reference,(B, self.n_groups, Hk, Wk, 2))


class TrajEncoder(nn.Module):
    def __init__(self,num_heads=4,out_dim=256):
        super(TrajEncoder, self).__init__()
        self.node_feature = nn.Sequential(nn.Conv1d(5, 64, kernel_size=1), nn.ELU())
        self.node_attention = MultiHeadAttention(input_channels=(64,64,64), num_heads=num_heads, head_size=64, dropout=0.1, output_size=64*5)
        self.vector_feature = nn.Linear(3, 64, bias=False)
        self.sublayer = nn.Sequential(nn.Linear(384, out_dim), nn.ELU())

    def forward(self, inputs, mask):
        mask = mask.to(torch.float32)
        mask = torch.matmul(mask[:, :, np.newaxis], mask[:, np.newaxis, :])
        nodes = self.node_feature(inputs[:, :, :5].permute(0,2,1))
        nodes = nodes.permute(0,2,1)
        nodes = self.node_attention(inputs=[nodes, nodes, nodes], mask=mask)
        nodes, _ = torch.max(nodes, 1)
        # vector = self.vector_feature(inputs[:, 0, 5:])
        vector = self.vector_feature(inputs[:, 0, 2:5])
        out = torch.concat([nodes, vector], dim=1)
        polyline_feature = self.sublayer(out)
        return polyline_feature


    
class Cross_Attention(nn.Module):
    def __init__(self, num_heads, key_dim, conv_attn=False):
        super(Cross_Attention, self).__init__()
        self.mha = MultiHeadAttention(input_channels=(key_dim, key_dim), num_heads=num_heads, head_size=key_dim//num_heads, output_size=key_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(eps=1e-3, normalized_shape=key_dim)
        self.norm2 = nn.LayerNorm(eps=1e-3, normalized_shape=key_dim)
        self.FFN1 = nn.Sequential(nn.Linear(key_dim, 4*key_dim),nn.ELU())
        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Sequential(nn.Linear(4*key_dim, key_dim),nn.ELU())
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, query, key, mask=None, training=True):
        value = self.mha(inputs=[query, key], mask=mask)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value)
        value = self.FFN2(value)
        value = self.dropout2(value)
        value = self.norm2(value)
        return value



class TrajNet(nn.Module):
    def __init__(self,cfg,past_to_current_steps=11,obs_actors=48,occ_actors=16,actor_only=True,no_attn=False,
        double_net=False):
        super(TrajNet, self).__init__()

        self.actor_only = actor_only
        self.obs_actors=obs_actors
        self.occ_actors=occ_actors
        self.double_net = double_net
        
        self.traj_encoder = TrajEncoder(num_heads=cfg['traj_heads'],out_dim=cfg['out_dim'])
        # self.traj_encoder  = TrajEncoderLSTM(cfg['out_dim'])
        self.no_attn = no_attn
        if not no_attn:
            if double_net:
                self.cross_attention = nn.Module([Cross_AttentionT(num_heads=cfg['att_heads'],key_dim=192,output_dim=cfg['out_dim']) for _ in range(2)])
            else:
                self.cross_attention = Cross_Attention(num_heads=cfg['att_heads'], key_dim=cfg['out_dim'])

        self.obs_norm = nn.LayerNorm(eps=1e-3, normalized_shape=cfg['out_dim'])
        self.occ_norm = nn.LayerNorm(eps=1e-3, normalized_shape=cfg['out_dim'])
        # self.obs_drop = tf.keras.layers.Dropout(0.1)
        # self.occ_drop = tf.keras.layers.Dropout(0.1)

        # dummy_obs_actors = torch.zeros([2,obs_actors,past_to_current_steps,8])
        # dummy_occ_actors = torch.zeros([2,occ_actors,past_to_current_steps,8])
        # dummy_ccl = tf.zeros([1,256,10,7])

        self.bi_embed = torch.tensor([[1,0],[0,1]], dtype=torch.float32).repeat_interleave(torch.tensor([obs_actors, occ_actors]), dim=0)
        self.seg_embed = nn.Linear(2, cfg['out_dim'], bias=False)

        # self(dummy_obs_actors,dummy_occ_actors)
        # summary(self)
    
    def forward(self,obs_traj,occ_traj,map_traj=None,training=True):

        obs_mask = torch.not_equal(obs_traj, 0)[:,:,:,0]
        num_obs = obs_traj.size()[1]
        obs = [self.traj_encoder(obs_traj[:, i],obs_mask[:,i]) for i in range(num_obs)]
        obs = torch.stack(obs,dim=1)
        num_occ = occ_traj.size()[1]
        occ_mask = torch.not_equal(occ_traj, 0)[:,:,:,0]
        occ = [self.traj_encoder(occ_traj[:, i],occ_mask[:,i]) for i in range(num_occ)]
        occ = torch.stack(occ,dim=1)

        embed = self.bi_embed[np.newaxis, :, :].repeat_interleave(occ.size()[0], dim=0).to(obs_traj.device)
        embed = self.seg_embed(embed)

        c_attn_mask = torch.not_equal(torch.max(torch.concat([obs_mask,occ_mask], dim=1).to(torch.int32),dim=-1)[0],0) #[batch,64] (last step denote the current)
        c_attn_mask = c_attn_mask.to(torch.float32)

        if self.no_attn:
            if self.double_net:
                concat_actors = torch.concat([obs,occ], dim=1)
                obs = self.obs_norm(concat_actors+embed)
                occ = self.occ_norm(concat_actors+embed)
                return obs,occ,c_attn_mask
            else:
                return self.obs_norm(obs + embed[:,:self.obs_actors,:]),self.occ_norm(occ + embed[:,self.obs_actors:,:]),c_attn_mask

        # interactions given seg_embedding
        concat_actors = torch.concat([obs, occ], dim=1)
        concat_actors = torch.multiply(c_attn_mask[:, :, np.newaxis].to(torch.float32), concat_actors)
        # query = concat_actors + embed
        query = concat_actors

        attn_mask = torch.matmul(c_attn_mask[:, :, np.newaxis], c_attn_mask[:, np.newaxis, :]) #[batch,64,64]

        if self.double_net:
            value = self.cross_attention[0](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs,val_occ = value[:,:self.obs_actors,:] , value[:,self.obs_actors:,:]

            value_flow = self.cross_attention[1](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs_f,val_occ_f = value_flow[:,:self.obs_actors,:] , value_flow[:,self.obs_actors:,:]

            obs = obs + val_obs
            occ = occ + val_occ

            ogm = torch.concat([obs,occ], dim=1) + embed

            obs_f = obs + val_obs_f
            occ_f = occ + val_occ_f

            flow = torch.concat([obs_f,occ_f], dim=1) + embed

            return self.obs_norm(ogm) , self.occ_norm(flow) , c_attn_mask
        
        value = self.cross_attention(query=query, key=concat_actors, mask=attn_mask, training=training)
        val_obs,val_occ = value[:,:num_obs,:] , value[:,num_obs:,:]

        obs = obs + val_obs
        occ = occ + val_occ

        concat_actors = torch.concat([obs,occ], dim=1)

        # obs = self.obs_norm(obs + embed[:,:self.obs_actors,:])
        # occ = self.occ_norm(occ + embed[:,self.obs_actors:,:])

        return obs,occ,c_attn_mask

class Cross_AttentionT(nn.Module):
    def __init__(self, num_heads, key_dim,output_dim,conv_attn=False,sep_actors=False):
        super(Cross_AttentionT, self).__init__()
        self.mha = MultiHeadAttention(input_channels=(key_dim * num_heads,key_dim * num_heads), num_heads=num_heads, head_size=key_dim//num_heads,output_size=key_dim,dropout=0.1)
        self.sep_actors = sep_actors
        
        self.norm1 = nn.LayerNorm(eps=1e3, normalized_shape=key_dim)
        self.norm2 = nn.LayerNorm(eps=1e3, normalized_shape=output_dim)
        self.FFN1 = nn.Sequential(nn.Linear(key_dim, 4*key_dim), nn.ELU())
        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Linear(4 * key_dim, output_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.conv_attn = conv_attn

    def forward(self, query, key, mask, training=True,actor_mask=None):
        value = self.mha(inputs=[query, key], mask=mask)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value)
        value = self.FFN2(value)
        value = self.dropout2(value)
        value = self.norm2(value)
        return value

class TrajNetCrossAttention(nn.Module):
    def __init__(self,traj_cfg,pic_size=(8,8),pic_dim=768,past_to_current_steps=11,obs_actors=48,occ_actors=16,actor_only=True,
        multi_modal=True,sep_actors=False):
        super(TrajNetCrossAttention, self).__init__()

        self.traj_net = TrajNet(traj_cfg,no_attn=traj_cfg['no_attn'],
            past_to_current_steps=past_to_current_steps,obs_actors=obs_actors,occ_actors=occ_actors,actor_only=actor_only,
            double_net=False)

        self.obs_actors = obs_actors
        self.H, self.W = pic_size
        self.pic_dim = pic_dim

        self.multi_modal = multi_modal
        self.actor_only = actor_only
        self.sep_actors = sep_actors
  
        self.cross_attn_obs = nn.ModuleList([Cross_AttentionT(num_heads=3, output_dim=pic_dim,key_dim=128,sep_actors=sep_actors) for _ in range(8)])

        # dummy_obs_actors = torch.zeros([2,obs_actors,past_to_current_steps,8])
        # dummy_occ_actors = torch.zeros([2,occ_actors,past_to_current_steps,8])
        # dummy_ccl = torch.zeros([2,256,10,7])
        # dummy_pic_encode = torch.zeros((2,) + pic_size + (pic_dim,))
        
        # flow_pic_encode = torch.zeros((2,) + pic_size + (pic_dim,))
        # if multi_modal:
        #     dummy_pic_encode = torch.zeros((2,8,) + pic_size + (pic_dim,))

        # self(dummy_pic_encode,dummy_obs_actors,dummy_occ_actors,dummy_ccl,flow_pic_encode=flow_pic_encode)
        # summary(self)
    
    def map_encode(self,map_traj,training):

        segs = map_traj.get_shape()[1]
        map_mask = torch.not_equal(map_traj[:,:,:,0], 0) #[batch,256,10]
        amap_mask = torch.reshape(map_mask,[-1,10])
        map_traj = torch.reshape(map_traj, [-1,10,7])
        map_enc = self.map_encoder(map_traj,amap_mask,training)
        map_enc = torch.reshape(map_enc,[-1,256,map_enc.get_shape()[-1]])

        map_mask = map_mask[:,:,0].to(torch.int32)#[batch,256]
        return map_enc,map_mask


    def forward(self,pic_encode,obs_traj,occ_traj,map_traj=None,training=True,flow_pic_encode=None):

        obs,occ,traj_mask = self.traj_net(obs_traj,occ_traj,map_traj,training)

        if self.sep_actors:
            actor_mask = torch.matmul(traj_mask[:, :, np.newaxis], traj_mask[:, np.newaxis, :])
        
        flat_encode = torch.reshape(pic_encode, shape=[-1,8,self.H*self.W,self.pic_dim])
        pic_mask = torch.ones_like(flat_encode[:,0,:,0],dtype=torch.float32)

        obs_attn_mask = torch.matmul(pic_mask[:, :, np.newaxis], traj_mask[:, np.newaxis, :])

        query = flat_encode
        key = torch.concat([obs,occ], dim=1)
        res_list = []
        for i in range(8):
            if self.sep_actors:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training,actor_mask)
            else:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training)
            v = o + query[:,i]
            res_list.append(v)
            
        obs_value = torch.stack(res_list,dim=1)
        obs_value = torch.reshape(obs_value, shape=[-1,8,self.H,self.W,self.pic_dim])

        return obs_value


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = F.elu(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * F.elu(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return 

class Pyramid3DDecoder(nn.Module):
    def __init__(self,config,img_size,pic_dim, use_pyramid=False,model_name='PyrDecoder',split_pred=False,
        timestep_split=False,double_decode=False,stp_grad=False,shallow_decode=0,flow_sep_decode=False,
        conv_cnn=False,sep_conv=False,rep_res=True,fg_sep=False):
        super(Pyramid3DDecoder, self).__init__()
        decode_inds = [4, 3, 2, 1, 0][shallow_decode:]
        decoder_channels = [48, 96, 128, 192, 384]

        self.stp_grad = stp_grad
        self.rep_res = rep_res

        #traj-rrc

        conv2d_kwargs = {
            'kernel_size': 3,
            'stride': 1,
            'padding': 'same',
        }

        self.upsample = [
            nn.Upsample(scale_factor=(1,2,2)) for i in decode_inds
        ]
        if conv_cnn:
            self.upconv_0s = nn.ModuleList([
                ConvLSTM(
                    input_dim=pic_dim,
                    hidden_dim=decoder_channels[decode_inds[0]] / 4,
                    num_layers=1,
                    return_all_layers=True,
                    batch_first=True,
                    **conv2d_kwargs
                )] + [
                    nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=decoder_channels[i], **conv2d_kwargs), 
                        nn.ELU()
                    ) for i in decode_inds[1:]
                ])
        else:
            self.upconv_0s = nn.ModuleList(
                nn.Sequential(nn.Conv2d(
                    in_channels=decoder_channels[i+shallow_decode],
                    out_channels=decoder_channels[i],
                    **conv2d_kwargs
                ), nn.ELU()) for i in decode_inds
            )
        self.flow_sep_decode = flow_sep_decode

        if flow_sep_decode:
            self.upsample_f = nn.ModuleList(
                nn.Upsample(scale_factor=(1,2,2)) for _ in decode_inds[-2:]
            )
            if sep_conv:
                self.upconv_f = nn.ModuleList(
                    ConvLSTM(
                        input_dim=pic_dim,
                        hidden_dim=96 / 4,
                        num_layers=1,
                        return_all_layers=True,
                        batch_first=True,
                        **conv2d_kwargs
                    ),
                    nn.Sequential(
                        nn.Conv2d(in_channels=96, out_channels=48, **conv2d_kwargs), 
                        nn.ELU()
                    )
                )
            else:
                self.upconv_f = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=decoder_channels[i+shallow_decode],
                            out_channels=decoder_channels[i],
                            **conv2d_kwargs
                        ), nn.ELU()
                    ) for i in decode_inds[-2:]
                ])
            self.res_f = nn.Sequential(
                nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(8,1,1), padding="same"), 
                nn.ELU()
            )

            self.output_layer_f = nn.Conv2d(
                in_channels=48,
                out_channels=2,
                **conv2d_kwargs)
        
        self.use_pyramid = use_pyramid
        if use_pyramid:
            self.res_layer = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=[96,192][i-2],
                            out_channels=decoder_channels[i],
                            kernel_size=(8,1,1),
                            stride=1,
                            padding="same"
                        ), nn.ELU()
                    ) for i in decode_inds[:3-shallow_decode]
                ])
            self.ind_list=[2,1,0][shallow_decode:]
            self.reshape_dim_h = [16,32,64][shallow_decode:]
            self.reshape_dim_w = [8,16,32][shallow_decode:]
        if flow_sep_decode:
            out_dim=2
        else:
            out_dim=4

        self.output_layer = nn.Conv2d(
            in_channels=48,
            out_channels=out_dim,
            **conv2d_kwargs)

    def get_flow_output(self,x):
        for upsample,uconv_0 in zip(self.upsample_f,self.upconv_f):
            x = x.permute(0,4,1,2,3)
            x = upsample(x)
            B, C, D, H, W = x.size()
            x = x.permute(0,2,1,3,4) # B, D, C, H, W
            x = x.reshape([-1, C, H, W])
            x = uconv_0(x)
            x = x.permute(0,2,3,1) # B*D, H, W, C
            x = x.reshape([B, D, H, W, -1]) # B, D, H, W, C
        
        B, D, H, W, C = x.size()
        x = x.permute(0,1,4,2,3) # B, D, C, H, W
        x = x.reshape([-1, C, H, W])
        x = self.output_layer_f(x)
        x = x.permute(0,2,3,1) # B*D, H, W, C
        x = x.reshape([B, D, H, W, -1])
        return x
    
    def forward(self,x,training=True,res_list=None):
        if self.stp_grad:
            x = x.detach()
        i = 0
        if self.flow_sep_decode:
            flow_res = res_list[0]
            res_list = res_list[1:]
        
        for upsample,uconv_0 in zip(self.upsample,self.upconv_0s):

            x = x.permute(0,4,1,2,3) # B, C, D, H, W
            x = upsample(x)
            B, C, D, H, W = x.size()
            x = x.permute(0,2,1,3,4) # change to B, D, C, H, W
            x = x.reshape([-1, C, H, W]) # B*D, C, H, W
            x = uconv_0(x)
            x = x.permute(0,2,3,1) # B*D, H, W, C
            x = x.reshape([B, D, H, W, -1]) # B, D, H, W, C

            if self.use_pyramid and i<=len(self.ind_list)-1:
                if self.rep_res:
                    res_flat  = torch.repeat_interleave(res_list[self.ind_list[i]][:,np.newaxis],repeats=8,dim=1)
                else:
                    res_flat = res_list[self.ind_list[i]]

                if self.stp_grad:
                    res_flat = res_flat.detach()
                h = res_flat.size()[-1]
                res_flat = torch.reshape(res_flat,[-1,8,self.reshape_dim_h[i],self.reshape_dim_w[i],h])
                res_flat = res_flat.permute(0,4,1,2,3) # B, C, D, H, W
                x = x + self.res_layer[i](res_flat).permute(0,2,3,4,1)

            if i==len(self.ind_list)-1 and self.flow_sep_decode:
                flow_res = torch.reshape(flow_res,[-1,64,32,96])
                flow_res = torch.repeat_interleave(flow_res[:,np.newaxis],repeats=8,dim=1)
                flow_res = flow_res.permute(0,4,1,2,3) # B, C, D, H, W
                flow_x = x + self.res_f(flow_res).permute(0,2,3,4,1)
            i+=1
        B, D, H, W, C = x.size()
        x = x.permute(0,1,4,2,3) # B, D, C, H, W
        x = x.reshape([-1, C, H, W]) # B*D, C, H, W
        x = self.output_layer(x)
        x = x.permute(0,2,3,1) # B*D, H, W, C
        x = x.reshape([B, D, H, W, -1])
        if self.flow_sep_decode:
            flow_x = self.get_flow_output(flow_x)
            x = torch.concat([x,flow_x],dim=-1)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,training=True):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.size()
    x = torch.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C])
    x = x.permute([0, 1, 3, 2, 4, 5])
    windows = torch.reshape(x, shape=[-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W, C):
    x = torch.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = x.permute([0, 1, 3, 2, 4, 5])
    x = torch.reshape(x, shape=[-1, H, W, C])
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # build
        self.relative_position_bias_table = nn.Parameter(data=torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads), requires_grad=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = nn.Parameter(data=torch.from_numpy(relative_position_index), requires_grad=False)
        self.built = True

    def forward(self, x, mask=None,training=False):
        B_, N, C = x.size()
        qkv = torch.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]).permute([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.permute([0, 1, 3, 2]))
        relative_position_bias = self.relative_position_bias_table[torch.reshape(
            self.relative_position_index, shape=[-1])]
        relative_position_bias = torch.reshape(relative_position_bias, shape=[
                                            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = relative_position_bias.permute([2, 0, 1])
        attn = attn + torch.unsqueeze(relative_position_bias, dim=0)

        if mask is not None:
            nW = mask.size()[0]  # tf.shape(mask)[0]
            attn = torch.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + torch.unsqueeze(torch.unsqueeze(mask, dim=1), dim=0).to(attn.dtype)
            attn = torch.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute([0, 2, 1, 3])
        x = torch.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=None):
        return self._drop_path(x)

    def _drop_path(self, inputs):
        if (not self.training) or (self.drop_prob == 0.):
            return inputs

        keep_prob = 1.0 - self.drop_prob
        
        shape = (inputs.size()[0],) + (1,) * (len(inputs.size()) - 1)
        
        random_tensor = keep_prob

        random_tensor += torch.rand(shape, dtype=inputs.dtype)
        binary_tensor = torch.floor(random_tensor).to(inputs.device)
        output = torch.divide(inputs, keep_prob) * binary_tensor
        return output

class SwinTransformerBlock(torch.nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=nn.LayerNorm, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = prefix

        self.norm1 = norm_layer(eps=1e-5, normalized_shape=dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=self.prefix)
        self.drop_path = DropPath(drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(eps=1e-5, normalized_shape=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix)

        # build
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = torch.from_numpy(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = torch.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = torch.unsqueeze(
                mask_windows, dim=1) - torch.unsqueeze(mask_windows, dim=2)
            attn_mask = torch.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = torch.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = nn.Parameter(data=attn_mask, requires_grad=False)
        else:
            self.attn_mask = None

        self.built = True

    def forward(self, x,training=False):
        H, W = self.input_resolution
        B, L, C = x.size()
        assert L == H * W, f"input feature has wrong size,{H},{W},{L},{H*W}"

        shortcut = x
        x = self.norm1(x)
        x = torch.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=[-self.shift_size, -self.shift_size], dims=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = torch.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask,training=training)

        # merge windows
        attn_windows = torch.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=[
                        self.shift_size, self.shift_size], dims=[1, 2])
        else:
            x = shifted_x
        x = torch.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x,training)
        
        x = x + self.drop_path(self.mlp(self.norm2(x),training),training)

        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(eps=1e-5, normalized_shape=4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.size()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = torch.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.concat([x0, x1, x2, x3], dim=-1)
        x = torch.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(torch.nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=torch.nn.LayerNorm, downsample=None, use_checkpoint=False, prefix=''):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(
                                               drop_path_prob, list) else drop_path_prob,
                                           norm_layer=norm_layer,
                                           prefix=f'{prefix}/blocks{i}') for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
        else:
            self.downsample = None

    def forward(self, x, training=False):
        for block in self.blocks:
            x = block(x,training)

        res = x

        if self.downsample is not None:
            x = self.downsample(x)
            return x,res
        else:
            return x,x

class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size,
                           stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(normalized_shape=embed_dim, eps=1e-5)
        else:
            self.norm = None

    def forward(self, x):
        B, H, W, C = x.size()

        x = x.permute(0,3,1,2) # B C H W
        x = self.proj(x)
        
        x = x.permute(0,2,3,1) # B H W C
        x = torch.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformerEncoder(torch.nn.Module):
    def __init__(self, include_top=False,
                 img_size=(224, 224), patch_size=(4, 4), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,sep_encode=False,no_map=False,flow_sep=False,use_flow=False,large_input=True, **kwargs):
        super(SwinTransformerEncoder, self).__init__()
        """
        Encoder of SwinTransformer

        Input:
        x : [batch, 256, 256, 10, 2] # 10s OGM of both vehicle(0) and cyclist_pedestrian(1)
        map_img : [batch, 256, 256, 3] # BEV map image

        Output:
        res_list: a list of results of SwinT Layer output:
        H = W = 255 , C = emb_dim
        [0-3]: 
        [H/4,W/4,C],
        [H/8,W/8,2C],
        [H/16,W/16,4C],
        [H/32,W/32,8C]
        """

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.flow_sep = flow_sep
        self.no_map=no_map

        self.use_flow = use_flow
        self.large_input = large_input

        # split image into non-overlapping patches
        self.patch_embed_vecicle = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=config.task.his_len, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
            
        self.sep_encode=sep_encode

        num_patches = self.patch_embed_vecicle.num_patches
        patches_resolution = self.patch_embed_vecicle.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        if sep_encode:
            
            if self.use_flow:
                self.patch_embed_flow = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None)
                
                if self.flow_sep:
                    self.flow_norm = norm_layer(eps=1e-5, normalized_shape=embed_dim )
                    self.flow_layer = BasicLayer(dim=int(embed_dim * (2 ** 0)),
                                                    input_resolution=(patches_resolution[0] // (2 ** 0),
                                                                    patches_resolution[1] // (2 ** 0)),
                                                    depth=depths[0],
                                                    num_heads=num_heads[0],
                                                    window_size=window_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path_prob=dpr[sum(depths[:0]):sum(
                                                        depths[:0 + 1])],
                                                    norm_layer=norm_layer,
                                                    downsample=PatchMerging if (
                                                        0 < self.num_layers - 1) else None,# No downsample of the last layer
                                                    use_checkpoint=use_checkpoint,
                                                    prefix=f'flow_layers{0}')      
            if not self.no_map:
                self.patch_embed_map = PatchEmbed(
                    img_size=(256,256), patch_size=patch_size, in_chans=3, embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None)
        # build layers
        self.basic_layers = nn.ModuleList([BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,# No downsample of the last layer
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)])
        
        self.final_resolution = patches_resolution[0] // (2 ** (self.num_layers - 1)), patches_resolution[1] // (2 ** (self.num_layers - 1))

        self.all_patch_norm = norm_layer(eps=1e-5, normalized_shape=embed_dim )
        
        # dummy_ogm = torch.zeros([1,img_size[0],img_size[1],11,2])
        # if self.large_input:
        #    dummy_map = torch.zeros([1,img_size[0]//2,img_size[1]//2,3]) 
        # else:
        #     dummy_map = torch.zeros([1,img_size[0],img_size[1],3])

        # dummy_flow =torch.zeros((1,)+(img_size[0],img_size[1])+(2,))

        # self(dummy_ogm,dummy_map,dummy_flow)
        # summary(self)

    def forward_features(self,x,map_img,flow=None,training=True):
        if self.sep_encode:
            vec = x
            # add batch dimension
            
            if self.no_map:
                x = self.patch_embed_vecicle(vec)
            elif self.flow_sep and self.use_flow:
                flow = self.patch_embed_flow(flow)
                flow = self.flow_norm(flow)
                flow_x,flow_res = self.flow_layer(flow,training)
                if not self.large_input:
                    x = self.patch_embed_vecicle(vec) +  self.patch_embed_map(map_img)
                else:
                    

                    maps = torch.reshape(maps,[-1,64,64,self.embed_dim])
                    maps = F.pad(maps,[0,0,32,32,32,32,0,0])
                    maps = torch.reshape(maps,[-1,128*128,self.embed_dim])
                    x = self.patch_embed_vecicle(vec)
                    x = x + maps
            else:
                if self.use_flow:
                    x = self.patch_embed_vecicle(vec) +  self.patch_embed_map(map_img) + self.patch_embed_flow(flow)
                else:
                    x = self.patch_embed_vecicle(vec) +  self.patch_embed_map(map_img)
        else:
            x = torch.reshape(x,[-1,256,256,11*2])
            if not self.no_map and self.use_flow:
                x = torch.concat([x,map_img,flow], dim=-1)
            elif not self.use_flow:
                x = torch.concat([x,map_img], dim=-1)
            x = self.patch_embed_vecicle(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.all_patch_norm(x)
        res_list=[]
        for i,st_layer in enumerate(self.basic_layers):
            x,res = st_layer(x,training)
            if i==self.num_layers - 1:
                H, W = self.final_resolution
                B, L, C = x.size()
                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                res = torch.reshape(res, shape=[-1, H, W, C])
            if i==0 and self.flow_sep and self.use_flow:
                x = x + flow_x
                if self.large_input:
                    flow_res = torch.reshape(torch.reshape(flow_res,[-1,128,128,self.embed_dim])[:,32:32+64,32:32+64,:],[-1,64*64,96])
                res_list.append(flow_res)
            if self.large_input:
                init_res = 128 // (2**i)
                dim = self.embed_dim * (2**i)
                crop = init_res // 2
                c_b,c_e = int(init_res*0.25),int(init_res*0.75)
                res = torch.reshape(torch.reshape(res,[-1,init_res,init_res,dim])[:,c_b:c_e,c_b:c_e,:],[-1,crop*crop,dim])
            res_list.append(res)
        return res_list

    def forward(self, x,map_img,flow,training=True):
        x = self.forward_features(x,map_img,flow,training)
        return x
    
class OFMPNet(torch.nn.Module):
    def __init__(self,cfg,use_pyramid=True,actor_only=True,sep_actors=False,
        fg_msa=False,use_last_ref=False,fg=False,large_ogm=False):

        super(OFMPNet, self).__init__()

        self.encoder = SwinTransformerEncoder(include_top=True,img_size=cfg['input_size'], window_size=cfg[
            'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads'],
            sep_encode=True,flow_sep=True,use_flow=True,drop_rate=0.0, attn_drop_rate=0.0,drop_path_rate=0.1,
            large_input=large_ogm)

         
        if sep_actors:
            traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=True)
        else:
            traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=False)
        
        resolution=[8,16,32]
        h = resolution[4-len(cfg['depths'][:])]
        resolution=[4,8,16]
        w = resolution[4-len(cfg['depths'][:])]
        self.trajnet_attn = TrajNetCrossAttention(traj_cfg,actor_only=actor_only,pic_size=(h,w),pic_dim=768//(2**(4-len(cfg['depths'][:])))
        ,multi_modal=True,sep_actors=sep_actors)
        self.fg_msa = fg_msa
        self.fg = fg
        if fg_msa:
            self.fg_msa_layer = FGMSA(q_size=(16,8), kv_size=(16,8),n_heads=8,n_head_channels=48,n_groups=8,out_dim=384,use_last_ref=False,fg=fg)
        self.decoder = Pyramid3DDecoder(config=None,img_size=cfg['input_size'],pic_dim=768//(2**(4-len(cfg['depths'][:]))),use_pyramid=use_pyramid,timestep_split=True,
        shallow_decode=(4-len(cfg['depths'][:])),flow_sep_decode=True,conv_cnn=False)

        # dummy_ogm =torch.zeros((1,)+cfg['input_size']+(11,2,))
        # dummy_map =torch.zeros((1,)+(256,256)+(3,))

        # dummy_obs_actors = torch.zeros([1,48,11,8])
        # dummy_occ_actors = torch.zeros([1,16,11,8])
        # dummy_ccl = torch.zeros([1,256,10,7])
        # dummy_flow =torch.zeros((1,)+cfg['input_size']+(2,))
        # self.ref_res = None

        # self(dummy_ogm,dummy_map,obs=dummy_obs_actors,occ=dummy_occ_actors,mapt=dummy_ccl,flow=dummy_flow)
        summary(self)
    
    def forward(self,input_dic,map_img,training=True,obs=None,occ=None,mapt=None,flow=None):
        
        #visual encoder:
        ogm_prv = input_dic['prv/state/his/observed_occupancy_map']# B T H W
        if ogm_prv.dim() == 3:
            ogm_prv = torch.unsqueeze(ogm_prv, 0)
        ogm_prv = ogm_prv.permute([0,2,3,1]) # B H W T
        
        flow_prv = input_dic['prv/state/his/flow_map']
        if flow_prv.dim() == 4:
            flow_prv = torch.unsqueeze(flow_prv, 0)
        flow_prv = flow_prv[:,0,:,:,:]# B H W 2
        
       

        ogm_cur = input_dic['cur/state/his/observed_occupancy_map']# B T H W
        if ogm_cur.dim() == 3:
            ogm_cur = torch.unsqueeze(ogm_cur, 0)
        ogm_cur = ogm_cur.permute([0,2,3,1]) # B H W T
        
        flow_cur = input_dic['cur/state/his/flow_map']
        if flow_cur.dim() == 4:
            flow_cur = torch.unsqueeze(flow_cur, 0)
        flow_cur = flow_cur[:,0,:,:,:]# B H W 2


            
        ogm_nxt = input_dic['nxt/state/his/observed_occupancy_map']# B T H W
        if ogm_nxt.dim() == 3:
            ogm_nxt = torch.unsqueeze(ogm_nxt, 0)
        ogm_nxt = ogm_nxt.permute([0,2,3,1]) # B H W T
        
        flow_nxt = input_dic['nxt/state/his/flow_map']
        if flow_nxt.dim() == 4:
            flow_nxt = torch.unsqueeze(flow_nxt, 0)
        flow_nxt = flow_nxt[:,0,:,:,:]# B H W 2

        
        
        q_prv = self.encoder(ogm_prv,map_img,flow_prv,training)[-1]

        res_list = self.encoder(ogm_cur,map_img,flow_cur,training)
        q_cur = res_list[-1]

        q_nxt = self.encoder(ogm_nxt,map_img,flow_nxt,training)[-1]

        q = q_cur + q_prv + q_nxt

        if self.fg_msa:
            q = torch.reshape(q,[-1,16,8,384])
            #fg-msa:
            res,pos,ref = self.fg_msa_layer(q,training=training)
            q = res + q
            q = torch.reshape(q,[-1,16*8,384])
        query = torch.repeat_interleave(torch.unsqueeze(q, dim=1),repeats=8,axis=1)
        if self.fg:
            # added Projected flow-features to each timestep
            ref = torch.reshape(ref,[-1,8,256,384])
            query = ref + query
        
        # traj_prv = torch.concatenate([input_dic['prv/his/x_position'][..., None],input_dic['prv/his/y_position'][..., None]],dim=-1).permute([0,2,1,3])
        traj_cur = torch.concatenate([input_dic['cur/state/his/x_position'][..., None],
                                      input_dic['cur/state/his/y_position'][..., None],
                                      input_dic['cur/state/his/x_velocity'][..., None],
                                      input_dic['cur/state/his/y_velocity'][..., None],
                                      input_dic['cur/state/his/yaw_angle'][..., None]]
                                     ,dim=-1)
        # traj_nxt = torch.concatenate([input_dic['nxt/his/x_position'][..., None],input_dic['nxt/his/y_position'][..., None]],dim=-1).permute([0,2,1,3])
        
       
        
        
        #time-sep-cross attention and vector encoders:
        obs_value = self.trajnet_attn(query,traj_cur,traj_cur,mapt,training)

        #fpn decoding:
        y = self.decoder(obs_value,training,res_list)
        y = torch.reshape(y.permute([0,2,3,1,4]),[-1,256,128,32])
        return y







