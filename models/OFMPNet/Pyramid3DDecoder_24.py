import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.OFMPNet.ConvLSTM import ConvLSTM
from torchinfo import summary

class Pyramid3DDecoder(nn.Module):
    def __init__(self, config):
        super(Pyramid3DDecoder, self).__init__()

        self.shallow_decode = config.model.Pyramid3DDecoder.shallow_decode
        self.pic_dim = config.model.Pyramid3DDecoder.pic_dim
        

        decode_inds = [4, 3, 2, 1, 0][self.shallow_decode:]
        # decoder_channels = [48, 96, 128, 192, 384]
        decoder_channels = [96, 128, 192, 384, 768]
        self.stp_grad = config.model.Pyramid3DDecoder.stp_grad
        self.rep_res = config.model.Pyramid3DDecoder.rep_res
        self.conv_cnn = config.model.Pyramid3DDecoder.conv_cnn
        self.flow_sep_decode = config.model.Pyramid3DDecoder.flow_sep_decode
        self.sep_conv = config.model.Pyramid3DDecoder.sep_conv
        self.use_pyramid = config.model.Pyramid3DDecoder.use_pyramid
        self.num_waypoints = config.task_config.num_waypoints
        #traj-rrc

        conv2d_kwargs = {
            'kernel_size': 3,
            'stride': 1,
            'padding': 'same',
        }

        self.upsample = [
            nn.Upsample(scale_factor=(1,2,2)) for i in decode_inds
        ]
        if self.conv_cnn:
            self.upconv_0s = nn.ModuleList([
                ConvLSTM(
                    input_dim=self.pic_dim,
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
                    in_channels=decoder_channels[i+self.shallow_decode],
                    out_channels=decoder_channels[i],
                    **conv2d_kwargs
                ), nn.ELU()) for i in decode_inds
            )
        

        if self.flow_sep_decode:
            self.upsample_f = nn.ModuleList(
                nn.Upsample(scale_factor=(1,2,2)) for _ in decode_inds[-2:]
            )
            if self.sep_conv:
                self.upconv_f = nn.ModuleList(
                    ConvLSTM(
                        input_dim=self.pic_dim,
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
                            in_channels=decoder_channels[i+self.shallow_decode],
                            out_channels=decoder_channels[i],
                            **conv2d_kwargs
                        ), nn.ELU()
                    ) for i in decode_inds[-2:]
                ])
            # self.res_f = nn.Sequential(
            #     nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(self.num_waypoints,1,1), padding="same"), 
            #     nn.ELU()
            # )
            self.res_f = nn.Sequential(
                nn.Conv3d(in_channels=192, out_channels=192, kernel_size=(self.num_waypoints,1,1), padding="same"), 
                nn.ELU()
            )
            # self.output_layer_f = nn.Conv2d(
            #     in_channels=48,
            #     out_channels=2,
            #     **conv2d_kwargs)
            
            self.output_layer_f = nn.Conv2d(
                in_channels=96,
                out_channels=2,
                **conv2d_kwargs)
        
        self.use_pyramid = self.use_pyramid
        # if self.use_pyramid:
        #     self.res_layer = nn.ModuleList([
        #             nn.Sequential(
        #                 nn.Conv3d(
        #                     in_channels=[96,192][i-2],
        #                     out_channels=decoder_channels[i],
        #                     kernel_size=(self.num_waypoints,1,1),
        #                     stride=1,
        #                     padding="same"
        #                 ), nn.ELU()
        #             ) for i in decode_inds[:3-self.shallow_decode]
        #         ])
        #     self.ind_list=[2,1,0][self.shallow_decode:]
        #     self.reshape_dim = [16,32,64][self.shallow_decode:]

        if self.use_pyramid:
            self.res_layer = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=[192,384][i-2],
                            out_channels=decoder_channels[i],
                            kernel_size=(self.num_waypoints,1,1),
                            stride=1,
                            padding="same"
                        ), nn.ELU()
                    ) for i in decode_inds[:3-self.shallow_decode]
                ])
            self.ind_list=[2,1,0][self.shallow_decode:]
            self.reshape_dim = [16,32,64][self.shallow_decode:]
        
        if self.flow_sep_decode:
            out_dim=2
        else:
            out_dim=4

        # self.output_layer = nn.Conv2d(
        #     in_channels=48,
        #     out_channels=out_dim,
        #     **conv2d_kwargs)
        self.output_layer = nn.Conv2d(
            in_channels=96,
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
    
    def forward(self,x, res_list=None,training=True,):
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
                    res_flat  = torch.repeat_interleave(res_list[self.ind_list[i]][:,np.newaxis],repeats=self.num_waypoints,dim=1)
                else:
                    res_flat = res_list[self.ind_list[i]]

                if self.stp_grad:
                    res_flat = res_flat.detach()
                h = res_flat.size()[-1]
                res_flat = torch.reshape(res_flat,[-1,self.num_waypoints,self.reshape_dim[i],self.reshape_dim[i],h])
                res_flat = res_flat.permute(0,4,1,2,3) # B, C, D, H, W
                x = x + self.res_layer[i](res_flat).permute(0,2,3,4,1)

            if i==len(self.ind_list)-1 and self.flow_sep_decode:
                flow_res = torch.reshape(flow_res,[-1,64,64,192])
                flow_res = torch.repeat_interleave(flow_res[:,np.newaxis],repeats=self.num_waypoints,dim=1)
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


from utils.file_utils import get_config
if __name__=='__main__':
    config = get_config('./config.yaml')
    model = Pyramid3DDecoder(config)
    