from __future__ import absolute_import, division, print_function
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils.block import Double_GRU, ConvBlock, DoubleBlock, Recurrent_unit

class Structure_up(nn.Module):
    def __init__(self, in_ch=2, nf=16, ndown = 4, nintermediate = 2,  
                 style_num = 0):
        super(Structure_up, self).__init__()
        init_layers = []
        init_layers +=[ConvBlock(in_ch, nf,
                             kernel_size = 3, stride=1, padding=1,
                             norm = 'none', activation = 'swish')] 
                       
        down_layers = []       
        for down in range(ndown):
            down_layers += [DoubleBlock(nf, nf*2, 
                              first_kernel=3, second_kernel= 3, 
                              stride=1, padding=1, style_num = style_num,
                              first_norm = 'none', second_norm = 'cin',
                              first_act= 'swish', second_act='swish')]
            down_layers += [nn.MaxPool2d(2)]
            nf = nf*2

        self.inter_layers = Recurrent_unit(nf,nf,kernel_size=3,
                                      norm = 'none', act = 'swish')

        #inter_layers = []
        #for inter in range(nintermediate):
        #    inter_layers += [Recurrent_unit(nf,nf,kernel_size=3, norm = 'none', act = 'swish')]
        #ConvBlock(nf, nf, kernel_size=3, stride=1, padding=1,  norm = 'in', activation='swish')]

        up_layers = [] 
        for up in range(ndown):
            up_layers += [nn.Upsample(scale_factor=2, mode='nearest')]

            up_layers += [DoubleBlock(nf*2, nf//2,
                                first_kernel=1, second_kernel= 3,
                                stride=1, padding=1,style_num = style_num,
                                first_norm = 'none', second_norm = 'cin',
                                first_act= 'swish', second_act='swish')]
            nf //=2       
             
        self.out_conv = ConvBlock(nf*2, 2, 
                           kernel_size = 3, stride = 1, padding = 1, 
                           norm='cin', activation='swish', 
                           style_num = style_num) 

        self.final_conv = ConvBlock(4, 1,
                           kernel_size= 3, stride= 1, padding= 1,
                           norm = 'none', activation = 'sigmoid')
        #self.output_conv = ConvBlock(1, 1,
        #                   kernel_size= 1, stride = 1, padding=0,
        #                   norm = 'none', activation = 'sigmoid')
               
        self.init_layers = nn.Sequential(*init_layers)
        self.down_layers = nn.Sequential(*down_layers)               
        #self.inter_layers = nn.Sequential(*inter_layers)
        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, in_x, style_id, hidden=None) :
        out = self.init_layers(in_x)

        down_list = list()
        down_list.append(out)
        for down_block in self.down_layers: 
            if isinstance(down_block, nn.MaxPool2d):
                out = down_block(out)
                down_list.append(out)
            else :
                out = down_block(out, style_id)

        #stats.reverse()
        down_list.reverse()
        out, hidden = self.inter_layers(out, hidden)

        i = 0
        for up_block in self.up_layers:
            if isinstance(up_block, nn.Upsample):
                skip = down_list[i]
                out = torch.cat([out, skip], 1)
                out = up_block(out)
                
                i += 1
            else:
                out = up_block(out, style_id)

        out = torch.cat([out, down_list[i]], 1)
        out = self.out_conv(out)
        out = torch.cat([out,in_x], 1)
        out = self.final_conv(out)
        #edge_out = in_x[:,1,:,:]*self.output_conv(out)
        #edge_out = self.output_conv(out)
        return out, hidden

class Structure_down(nn.Module):
    def __init__(self, in_ch=2, nf=16, ndown = 4, nintermediate=2, 
                 style_num = 0):
        super(Structure_down, self).__init__()
        init_layers = []
        init_layers +=[ConvBlock(in_ch, nf,
                             kernel_size = 3, stride=1, padding=1,
                             norm = 'none', activation = 'swish')]

        down_layers = []
        for down in range(ndown):
            down_layers +=[DoubleBlock(nf, nf*2,
                              first_kernel=3, second_kernel= 3,
                              stride=1, padding=1,style_num = style_num,
                              first_norm = 'none', second_norm='cin',
                              first_act= 'swish', second_act='swish')]
            down_layers += [nn.MaxPool2d(2)]
            nf = nf*2

        self.inter_layers = Recurrent_unit(nf,nf,kernel_size=3,
                                      norm = 'none', act = 'swish')
        #inter_layers = []
        #for inter in range(nintermediate):
        #    inter_layers += [Recurrent_unit(nf,nf,kernel_size=3, norm = 'none', act = 'swish')]
            #ConvBlock(nf, nf,kernel_size=3, stride=1, padding=1,norm = 'in', activation='swish')]

        up_layers = []
        for up in range(ndown):
            up_layers += [nn.Upsample(scale_factor=2, mode='nearest')]

            up_layers += [DoubleBlock(nf*2, nf//2,
                              first_kernel=1, second_kernel= 3,
                              stride=1, padding=1, style_num=style_num,
                              first_norm = 'none', second_norm='cin',
                              first_act= 'swish', second_act='swish')]
            nf //=2

        self.out_conv = ConvBlock(nf*2, 2,
                           kernel_size = 3, stride = 1, padding =1,
                           norm='cin', activation='swish', style_num = style_num)

        self.final_conv = ConvBlock(4,1,
                           kernel_size= 3, stride= 1, padding =1,
                           norm = 'none', activation = 'sigmoid')
        #self.output_conv = ConvBlock(1,1,
        #                   kernel_size= 1, stride = 1, padding=0,
        #                   norm = 'none', activation = 'sigmoid')

        self.init_layers = nn.Sequential(*init_layers)
        self.down_layers = nn.Sequential(*down_layers)
#        self.inter_layers = nn.Sequential(*inter_layers)
        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, in_x, style_id, hidden = None) :
        out = self.init_layers(in_x)
        
        #stats = list()
        down_list = list()
        down_list.append(out)
        for down_block in self.down_layers:
            if isinstance(down_block, nn.MaxPool2d):
                out = down_block(out)
                down_list.append(out)
            else :
                out = down_block(out, style_id)

        down_list.reverse()
        out, hidden = self.inter_layers(out, hidden)

        i = 0
        for up_block in self.up_layers:
            if isinstance(up_block, nn.Upsample):
                skip = down_list[i]
                out = torch.cat([out, skip], 1)
                out = up_block(out)

                i += 1
            else:
                out = up_block(out, style_id)

        out = torch.cat([out, down_list[i]], 1)
        out = self.out_conv(out, style_id)

        out = torch.cat([out,in_x], 1)
        out = self.final_conv(out)

        #edge_out = self.output_ddconv(out)
        #edge_out = in_x[:,1,:,:]*self.output_conv(out)

        return out, hidden

class CombineNet_up(nn.Module):
    def __init__(self, in_ch=2, nf=16, ndown = 4, nintermediate = 2):
        super(CombineNet_up, self).__init__()
        init_layers = []
        init_layers +=[ConvBlock(in_ch, nf,
                             kernel_size = 3, stride=1, padding=1,
                             norm = 'none', activation = 'swish')]

        down_layers = []
        for down in range(ndown):
            down_layers +=[DoubleBlock(nf, nf*2,
                              first_kernel=3, second_kernel= 3,
                              stride=1, padding=1,
                              first_norm = 'none', second_norm = 'none',
                              first_act= 'swish', second_act='swish')]
            down_layers += [nn.AvgPool2d(2)]
            nf = nf*2

        inter_layers = []
        for inter in range(nintermediate):
            inter_layers += [ConvBlock(nf, nf,kernel_size=3, stride=1, padding=1, norm = 'none', activation='swish')]

        up_layers = []
        for up in range(ndown):
            up_layers += [nn.Upsample(scale_factor=2, mode='bilinear')]

            up_layers += [DoubleBlock(nf*2, nf//2,
                                first_kernel=3, second_kernel= 3,
                                stride=1, padding=1,
                                first_norm = 'none', second_norm = 'in',
                                first_act= 'swish', second_act='swish')]
            nf //=2

        self.out_conv = ConvBlock(nf*2, 3,
                           kernel_size = 3, stride = 1, padding = 1,
                           norm='none', activation='tanh')

        #self.final_conv = ConvBlock(in_ch + 3, 3,
        #                   kernel_size= 3, stride= 1, padding= 1,
        #                   norm = 'none', activation = 'tanh')

        self.init_layers = nn.Sequential(*init_layers)
        self.down_layers = nn.Sequential(*down_layers)
        self.inter_layers = nn.Sequential(*inter_layers)
        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, in_x) :
        out = self.init_layers(in_x)

        down_list = list()
        down_list.append(out)
        for down_block in self.down_layers:
            if isinstance(down_block, nn.AvgPool2d):
                out = down_block(out)
                down_list.append(out)
            else :
                out = down_block(out)

        #stats.reverse()
        down_list.reverse()
        out = self.inter_layers(out)

        i = 0
        for up_block in self.up_layers:
            if isinstance(up_block, nn.Upsample):
                skip = down_list[i]
                out = torch.cat([out, skip], 1)
                out = up_block(out)

                i += 1
            else:
                out = up_block(out)

        out = torch.cat([out, down_list[i]], 1)
        out = self.out_conv(out)
        out = (out+in_x[:,3:6,:])/2
        
        #out = torch.cat([in_x,out], 1)
        #out = self.final_conv(out)

        return out

class CombineNet_down(nn.Module):
    def __init__(self, in_ch=2, nf=16, ndown = 4, nintermediate = 2):
        super(CombineNet_down, self).__init__()
        init_layers = []
        init_layers +=[ConvBlock(in_ch, nf,
                             kernel_size = 3, stride=1, padding=1,
                             norm = 'none', activation = 'swish')]

        down_layers = []
        for down in range(ndown):
            down_layers +=[DoubleBlock(nf, nf*2,
                              first_kernel=3, second_kernel= 3,
                              stride=1, padding=1,
                              first_norm = 'none', second_norm = 'none',
                              first_act= 'swish', second_act='swish')]
            down_layers += [nn.AvgPool2d(2)]
            nf = nf*2

        inter_layers = []
        for inter in range(nintermediate):
            inter_layers += [ConvBlock(nf, nf,kernel_size=3, stride=1, padding=1, norm = 'none', activation='swish')]

        up_layers = []
        for up in range(ndown):
            up_layers += [nn.Upsample(scale_factor=2, mode='bilinear')]

            up_layers += [DoubleBlock(nf*2, nf//2,
                                first_kernel=3, second_kernel= 3,
                                stride=1, padding=1,
                                first_norm = 'none', second_norm = 'in',
                                first_act= 'swish', second_act='swish')]
            nf //=2

        self.out_conv = ConvBlock(nf*2, 3,
                           kernel_size = 3, stride = 1, padding = 1,
                           norm='none', activation='tanh')

        #self.final_conv = ConvBlock(in_ch+3, 3,
        #                   kernel_size= 3, stride= 1, padding= 1,
        #                   norm = 'none', activation = 'tanh')

        self.init_layers = nn.Sequential(*init_layers)
        self.down_layers = nn.Sequential(*down_layers)
        self.inter_layers = nn.Sequential(*inter_layers)
        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, in_x) :
        out = self.init_layers(in_x)

        down_list = list()
        down_list.append(out)
        for down_block in self.down_layers:
            if isinstance(down_block, nn.AvgPool2d):
                out = down_block(out)
                down_list.append(out)
            else :
                out = down_block(out)

        #stats.reverse()
        down_list.reverse()
        out = self.inter_layers(out)

        i = 0
        for up_block in self.up_layers:
            if isinstance(up_block, nn.Upsample):
                skip = down_list[i]
                out = torch.cat([out, skip], 1)
                out = up_block(out)

                i += 1
            else:
                out = up_block(out)

        out = torch.cat([out, down_list[i]], 1)
        out = self.out_conv(out)
        out = (out + in_x[:,3:6,:])/2
        #out = torch.cat([in_x, out], 1)
        #out = self.final_conv(out)
        return out



