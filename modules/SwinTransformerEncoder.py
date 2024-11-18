import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


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
    def __init__(self, config,):
        super(SwinTransformerEncoder, self).__init__()
        """
        Encoder of SwinTransformer

        Input:
        x : [batch, 256, 256, his_len, 1] # his_len OGM of both vehicle
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

        # if the occ map is larger than road map
        self.large_input= config.model.swin_transformer.large_input
        self.transformer_depths = config.model.transformer_depths
        self.attention_heads = config.model.swin_transformer.attention_heads
        self.num_layers = len(self.transformer_depths)
        self.history_length = config.model.history_length
        self.input_size = config.model.input_size
        self.embedding_dimension = config.model.swin_transformer.embedding_dimension
        self.ape = config.model.swin_transformer.ape
        self.patch_norm = config.model.swin_transformer.patch_norm
        self.num_features = int(self.embedding_dimension * 2 ** (self.num_layers - 1))
        self.mlp_ratio = config.model.swin_transformer.mlp_ratio
        self.flow_sep = config.model.swin_transformer.flow_sep
        self.no_map = config.model.swin_transformer.no_map

        self.use_flow = config.model.swin_transformer.use_flow
        self.large_input = config.model.swin_transformer.large_input
        self.patch_size = config.model.swin_transformer.patch_size
        self.window_size = config.model.swin_transformer.window_size
        self.norm_layer = torch.nn.LayerNorm
        self.use_checkpoint = config.model.swin_transformer.use_checkpoint
        self.drop_path_rate = config.model.swin_transformer.drop_path_rate

        # self.qk_scale = config.model.swin_transformer.basic_layer.qk_scale
        self.qk_scale = None
        self.qkv_bias = config.model.swin_transformer.basic_layer.qkv_bias
        self.drop_rate = config.model.swin_transformer.basic_layer.drop_rate
        self.attn_drop_rate = config.model.swin_transformer.basic_layer.attn_drop_rate
        # split image into non-overlapping patches
        self.patch_embed_vecicle = PatchEmbed(
            img_size=self.input_size, patch_size=self.patch_size, in_chans=self.history_length, embed_dim=self.embedding_dimension,
            norm_layer=self.norm_layer if self.patch_norm else None)
            
        self.sep_encode=config.model.swin_transformer.sep_encode

        num_patches = self.patch_embed_vecicle.num_patches
        patches_resolution = self.patch_embed_vecicle.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embedding_dimension))

        dpr = [x for x in np.linspace(0., self.drop_path_rate, sum(self.transformer_depths))]

        if self.sep_encode:
            
            if self.use_flow:
                self.patch_embed_flow = PatchEmbed(
                    img_size=self.input_size, patch_size=self.patch_size, in_chans=2, embed_dim=self.embedding_dimension,
                    norm_layer=self.norm_layer if self.patch_norm else None)
                
                if self.flow_sep:
                    self.flow_norm = self.norm_layer(eps=1e-5, normalized_shape=self.embedding_dimension )
                    self.flow_layer = BasicLayer(dim=int(self.embedding_dimension * (2 ** 0)),
                                                    input_resolution=(patches_resolution[0] // (2 ** 0),
                                                                    patches_resolution[1] // (2 ** 0)),
                                                    depth=self.transformer_depths[0],
                                                    num_heads=self.attention_heads[0],
                                                    window_size=self.window_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                                    drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                                    drop_path_prob=dpr[sum(self.transformer_depths[:0]):sum(
                                                        self.transformer_depths[:0 + 1])],
                                                    norm_layer=self.norm_layer,
                                                    downsample=PatchMerging if (
                                                        0 < self.num_layers - 1) else None,# No downsample of the last layer
                                                    use_checkpoint=self.use_checkpoint,
                                                    prefix=f'flow_layers{0}')      
            if not self.no_map:
                self.patch_embed_map = PatchEmbed(
                    img_size=self.input_size, patch_size=self.patch_size, in_chans=3, embed_dim=self.embedding_dimension,
                    norm_layer=self.norm_layer if self.patch_norm else None)
        # build layers
        self.basic_layers = nn.ModuleList([BasicLayer(dim=int(self.embedding_dimension * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=self.transformer_depths[i_layer],
                                                num_heads=self.attention_heads[i_layer],
                                                window_size=self.window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                                drop_path_prob=dpr[sum(self.transformer_depths[:i_layer]):sum(
                                                    self.transformer_depths[:i_layer + 1])],
                                                norm_layer=self.norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,# No downsample of the last layer
                                                use_checkpoint=self.use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)])
        
        self.final_resolution = patches_resolution[0] // (2 ** (self.num_layers - 1)), patches_resolution[1] // (2 ** (self.num_layers - 1))

        self.all_patch_norm = self.norm_layer(eps=1e-5, normalized_shape=self.embedding_dimension )
        
        # dummy_ogm = torch.zeros([1,img_size[0],img_size[1],11,2])
        # if self.large_input:
        #    dummy_map = torch.zeros([1,img_size[0]//2,img_size[1]//2,3]) 
        # else:
        #     dummy_map = torch.zeros([1,img_size[0],img_size[1],3])

        # dummy_flow =torch.zeros((1,)+(img_size[0],img_size[1])+(2,))

        # self(dummy_ogm,dummy_map,dummy_flow)
        # summary(self)

    def forward_features(self, occupancy_map, flow_map=None, road_map=None, training=True):
        if self.sep_encode:
            
            
            if self.no_map:
                x = self.patch_embed_vecicle(occupancy_map)
            elif self.flow_sep and self.use_flow:
                flow_map = self.patch_embed_flow(flow_map)
                flow_map = self.flow_norm(flow_map)
                flow_x, flow_res = self.flow_layer(flow_map,training)
                if not self.large_input:
                    x = self.patch_embed_vecicle(occupancy_map) +  self.patch_embed_map(road_map)
                else:
                    maps = self.patch_embed_map(road_map)
                    # B, H, W, C -> B, H/patch_size, W/patch_size, embed_dim
                    maps = torch.reshape(maps,[-1,64,64,self.self.embedding_dimension])
                    # B, H, W, C -> B, +32 H +32, +32 W +32, C
                    maps = F.pad(maps,[0,0,32,32,32,32,0,0])
                    maps = torch.reshape(maps,[-1,128*128,self.self.embedding_dimension])
                    x = self.patch_embed_vecicle(occupancy_map)
                    x = x + maps
            else:
                if self.use_flow:
                    x = self.patch_embed_vecicle(occupancy_map) +  self.patch_embed_map(road_map) + self.patch_embed_flow(flow_map)
                else:
                    x = self.patch_embed_vecicle(occupancy_map) +  self.patch_embed_map(road_map)
        else:
            x = torch.reshape(occupancy_map,[-1,*self.img_size,self.his_len*2])
            if not self.no_map and self.use_flow:
                x = torch.concat([x,road_map, flow_map], dim=-1)
            elif not self.use_flow:
                x = torch.concat([x,road_map], dim=-1)
            x = self.patch_embed_vecicle(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.all_patch_norm(x)
        res_list=[]
        for i,st_layer in enumerate(self.basic_layers):
            # X : Downsampled feature, res : Original feature
            
            x, res = st_layer(x,training)
            # last layer
            if i==self.num_layers - 1:
                H, W = self.final_resolution
                B, L, C = x.size()
                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                res = torch.reshape(res, shape=[-1, H, W, C])
            # First layer
            if i==0 and self.flow_sep and self.use_flow:
                x = x + flow_x
                if self.large_input:
                    flow_res = torch.reshape(torch.reshape(flow_res,[-1,128,128,self.self.embedding_dimension])[:,32:32+64,32:32+64,:],[-1,64*64,96])
                res_list.append(flow_res)
            if self.large_input:
                init_res = 128 // (2**i)
                dim = self.self.embedding_dimension * (2**i)
                crop = init_res // 2
                c_b,c_e = int(init_res*0.25),int(init_res*0.75)
                res = torch.reshape(torch.reshape(res,[-1,init_res,init_res,dim])[:,c_b:c_e,c_b:c_e,:],[-1,crop*crop,dim])
            res_list.append(res)
        return res_list

    def forward(self, occupancy_map, road_map, flow_map, training=True):
        res_list = self.forward_features(occupancy_map, road_map, flow_map, training)
        return res_list
from utils.file_utils import get_config

if __name__=='__main__':
    config = get_config('./config.yaml')
    model = SwinTransformerEncoder(config)
    occupancy_map = torch.randn((2,256,256,40))
    road_map = torch.randn((2,256,256,3))
    flow_map = torch.randn((2,256,256,40, 2))
    res = model(occupancy_map,road_map,flow_map[:,:,:,-1,:])
