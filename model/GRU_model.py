from __future__ import absolute_import, division, print_function
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T

from utils.block import Double_GRU, ConvBlock, DoubleBlock, Recurrent_unit 

class Generator_up(nn.Module):
    def __init__(self, in_ch=3, nf=64, ndown = 5, nintermediate = 1, style_num=0):
        super(Generator_up, self).__init__()
        init_layers = []
        init_layers +=[ConvBlock(in_ch, nf,
                             kernel_size = 3, stride=1, padding=1,
                             norm = 'none', activation = 'swish')] 
                       
        down_layers = []       
        for down in range(ndown):            
            down_layers +=[DoubleBlock(nf, nf*2, 
                            first_kernel=3, second_kernel=3, 
                            stride=1, padding=1, 
                            first_norm = 'none', second_norm='none',
                            first_act = 'swish', second_act='swish')]

            down_layers += [nn.AvgPool2d(2)]
            nf = nf*2

        #inter_layers = []
        #for inter in range(nintermediate):
        #    inter_layers += [Recurrent_unit(nf,nf,kernel_size=3, 
        #                     norm = 'none', act = 'swish')]
            #inter_layers += [ConvGRUCell(nf, nf, kernel_size=3)]
            #ConvBlock(nf, nf, kernel_size=3, stride =1, padding=1, norm = 'none', activation = 'swish')]

        self.inter_layers = Recurrent_unit(nf,nf,kernel_size=3,
                                      norm = 'none', act = 'swish')

        up_layers = [] 
        for up in range(ndown):
            up_layers += [nn.Upsample(scale_factor=2, mode='bilinear')]
            up_layers += [DoubleBlock(nf*2, nf//2, 
                          first_kernel = 1, second_kernel=3,
                          stride = 1, padding= 1, style_num = style_num,
                          first_norm ='none', second_norm = 'cin', 
                          first_act= 'swish', second_act='swish')]
            nf //=2       
        
        
        self.out_conv = ConvBlock(nf*2, 3, 
                           kernel_size = 3, stride = 1, padding = 1, 
                           norm = 'none', activation='tanh')
        #self.final_conv = ConvBlock(6,3,
        #                   kernel_size=3, stride=1, padding= 1,
        #                   norm = 'none', activation = 'tanh')
                            
       
        self.init_layers = nn.Sequential(*init_layers)
        self.down_layers = nn.Sequential(*down_layers)               

        #self.inter_layers = nn.Sequential(*inter_layers)
        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, in_x, style_id = 0,  hidden=None) :
        out = self.init_layers(in_x)
        
        #stats = list()
        down_list = list()
        down_list.append(out)
        for down_block in self.down_layers: 
            #out, mean, std = down_block(out)          
            #stats.append((mean,std))
            if isinstance(down_block, nn.AvgPool2d):
                out = down_block(out)
                down_list.append(out)
            else:
                out = down_block(out)
        

        #stats.reverse()
        down_list.reverse()
        out, hidden = self.inter_layers(out, hidden)

        i = 0
        for up_block in self.up_layers:
            if isinstance(up_block, nn.Upsample):
                #beta, gamma = stats[i]
                #out = out*gamma + beta
                skip = down_list[i]
                out = torch.cat([out, skip], 1)
                out = up_block(out)
                
                i += 1
            else:
                out = up_block(out, style_id)
        
        out = torch.cat([out,down_list[i]], 1)
        out = self.out_conv(out)
        out = (out+in_x)/2
        #out = torch.cat([in_x,out], 1)
        #out = self.final_conv(out)

        return out, hidden

    def masking_up(self, prediction, in_x):
        thresh = 0.95
        slope = 4

        denorm_x = in_x/2 +0.5
        denorm_x = torch.clamp(denorm_x, 0,1)

        alpha = torch.where(denorm_x>thresh,
                            (1-denorm_x)/slope,
                            (slope-2*(1-thresh))/(slope*(1-2*thresh))*(denorm_x-thresh)+(1-thresh)/slope)
        alpha = torch.where(denorm_x <(1-thresh),
                            -(alpha-(1-thresh))/slope+(1-thresh/slope),
                            alpha)

        #prediction = torch.exp(prediction)
        output = torch.cat([alpha,prediction],1)
        return output
        
class Generator_down(nn.Module):
    def __init__(self, in_ch=3, nf=64, ndown = 5, nintermediate=1, style_num=0):
        super(Generator_down, self).__init__()
        init_layers = []
        init_layers +=[ConvBlock(in_ch, nf,
                             kernel_size = 3, stride=1, padding=1,
                             norm = 'none', activation = 'swish')]

        down_layers = []
        for down in range(ndown):

            down_layers +=[DoubleBlock(nf, nf*2,
                               first_kernel =3, second_kernel=3,
                               stride=1, padding=1, 
                               first_norm = 'none', second_norm ='none',
                               first_act = 'swish', second_act='swish')]

            down_layers += [nn.AvgPool2d(2)]
            nf = nf*2

        self.inter_layers = Recurrent_unit(nf,nf,kernel_size=3,
                                      norm = 'none', act = 'swish')
        #inter_layers = []
        #for inter in range(nintermediate):
        #    inter_layers += [Recurrent_unit(nf,nf,kernel_size=3, norm = 'none', act = 'swish')]

            #inter_layers += [ConvGRUCell(nf, nf, kernel_size=3)]
            #ConvBlock(nf, nf, kernel_size=3, stride =1, padding=1, norm = 'none'  activation = 'swish')]

        up_layers = []
        for up in range(ndown):
            up_layers += [nn.Upsample(scale_factor=2, mode='bilinear')]
            up_layers += [DoubleBlock(nf*2, nf//2,
                          first_kernel = 1, second_kernel=3,
                          stride = 1, padding= 1, style_num = style_num,
                          first_norm = 'none', second_norm = 'cin',
                          first_act = 'swish', second_act = 'swish')]
            nf //=2


        self.out_conv = ConvBlock(nf*2, 3,
                           kernel_size = 3, stride = 1, padding = 1,
                           norm='none', activation='tanh')
        #self.final_conv = ConvBlock(6,3,
        #                   kernel_size=3, stride=1, padding= 1,
        #                   norm = 'none', activation = 'tanh')

        self.init_layers = nn.Sequential(*init_layers)
        self.down_layers = nn.Sequential(*down_layers)
        #self.inter_layers = nn.Sequential(*inter_layers)
        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, in_x,style_id = 0, hidden = None) :
        out = self.init_layers(in_x)
        
        #stats = list()
        down_list = list()
        down_list.append(out)
        for down_block in self.down_layers:
            #out, mean, std = down_block(out)
             #stats.append((mean,std))
            if isinstance(down_block, nn.AvgPool2d):
                out = down_block(out)
                down_list.append(out)
            else :
                out = down_block(out)

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

        out = torch.cat([out,down_list[i]], 1)
        out = self.out_conv(out, style_id)
        #out = torch.cat([in_x, out], 1)
        #out = self.final_conv(out)
        out = (out+in_x)/2
        return out, hidden


    def masking_down(self, prediction, in_x):
        thresh = 0.05
        slope = 4
        denorm_x = in_x/2 + 0.5
        denorm_x = torch.clamp(denorm_x, 0,1)
        #alpha,_ = torch.max(denorm_x, dim= 1)

        alpha = torch.where(denorm_x < thresh,
                            (thresh-denorm_x)/slope,
                             (1-2*thresh/slope)/((1-2*thresh))*(denorm_x-thresh) + thresh/slope)
        alpha = torch.where(denorm_x> (1-thresh),
                            (alpha-(1-thresh))/slope+(1-thresh/slope),
                            alpha)

        output = torch.cat([alpha,prediction], 1) 
        return output


class discrim(nn.Module):
    def __init__(self, in_ch=6, nf=64):
        super(discrim, self).__init__()
        n_downs = 4

        layers= []
        layers.append(ConvBlock(in_ch, nf,
                               kernel_size=3, stride=1,
                               padding =1, norm='none',
                               activation='none'))
        layers += [nn.AvgPool2d(2)]

        dim = nf
        for n_down in range(n_downs):
            layers += [ConvBlock(dim, dim*2,
                               kernel_size = 3, stride=1,
                               padding =1, norm = 'none',
                               activation = 'leaky')]
            layers += [nn.AvgPool2d(2)]
            dim = dim*2

        self.body_net = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(dim, 1,
                              kernel_size = 3, stride =1,
                              padding = 1, bias=False)

    def forward(self, in_x, pred_ev):
        in_x = torch.cat([in_x, pred_ev], 1)

        output = self.body_net(in_x)
        output = self.conv1(output)
        return output



    

