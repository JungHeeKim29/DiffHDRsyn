import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import functools
from torch.autograd import Function
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.transforms as T
import scipy.io as sio

#########################################################
def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        return x*self.sigmoid(x)

class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=False, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std

class MS(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MS, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x

class batch_InstanceNorm2d(torch.nn.Module):
    def __init__(self, style_num, in_channels):
        super(batch_InstanceNorm2d, self).__init__()        
        self.inns = torch.nn.ModuleList([nn.InstanceNorm2d(in_channels, 
                                      affine=True) for i in range(style_num)])

    def forward(self, in_x, style_id):
        out = self.inns[style_id](in_x)        
        return out



############################################################################
class Preact_ConvBlock(nn.Module):
    def __init__(self, input_channels,output_channels, 
                 kernel_size, stride, padding, dilation=1, 
                 norm = 'spec', activation='relu'):
        super(Preact_ConvBlock, self).__init__()
         
        self.conv = nn.Conv2d(input_channels, 
                              output_channels,
                              kernel_size, 
                              stride, 
                              padding=padding)
        self.norm_name = norm 
        self.act_name = activation

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(input_channels)
            self.conv = nn.Conv2d(input_channels, output_channels,
                                  kernel_size, stride, padding = padding,
                                  bias = False)
                                  
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(input_channels)

        elif norm == 'spec':
            self.norm = nn.utils.spectral_norm()
        elif norm == 'none':
            self.norm = Identity()
        elif norm == 'pono':
            self.norm  = nn.InstanceNorm2d(input_channels,
                                          affine=False,
                                          track_running_stats=False)
            self.norm2 = PONO(affine=False)
        elif norm == 'ms':
            self.norm = MS()
        elif norm == 'group':
            self.norm = nn.GroupNorm(1,input_channels)

        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.01, True)
        elif activation == 'swish':
            self.act = Swish()
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'none':
            self.act = Identity()        

    def forward(self, x, beta=None, gamma=None):
        mean, std = None, None
        if self.norm_name == 'ms':
            x = self.norm(x, beta,gamma)
        else : 
            x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        if self.norm_name == 'pono':
            x, mean, std = self.norm2(x)
        if mean is None:
            return x
        else:
            return x, mean, std
    
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, 
                 kernel_size, stride, padding, dilation=1, 
                 norm = 'spec',activation='relu', style_num =0):

        super(ConvBlock, self).__init__()
       
        self.conv = nn.Conv2d(input_channels,output_channels,
                              kernel_size, stride, 
                              padding=padding)        
        self.norm_name = norm
        self.act_name = activation

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_channels)
            self.conv = nn.Conv2d(input_channels, output_channels,
                                kernel_size, stride, padding = padding,
                                bias = False)
 
        elif norm == 'in':
            self.norm =  nn.InstanceNorm2d(output_channels,
                                           affine= True,
                                           track_running_stats=False)
        elif norm == 'spec':
            self.norm = nn.utils.spectral_norm()
        elif norm == 'none':
            self.norm = Identity()

        elif norm == 'pono':
            self.norm = PONO()
            self.norm2 = nn.BatchNorm2d(output_channels)
        elif norm == 'ms':
            self.norm = MS()
        elif norm == 'cin':
            self.norm = batch_InstanceNorm2d(style_num, output_channels)

        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.01, True)
        elif activation == 'swish':
            self.act = Swish()
        elif activation == 'none':
            self.act = Identity()

    def forward(self, x, style_id = 0,  beta=None,gamma=None):
        mean, std = None, None

        x = self.conv(x)
        if self.norm_name == 'pono':
            x, mean, std = self.norm(x)
            x = self.norm2(x)

        elif self.norm_name == 'ms':
            x = self.norm(x, beta,gamma)
        elif self.norm_name == 'cin':
            x = self.norm(x, style_id)

        else :
            x = self.norm(x)

        x = self.act(x)

        if mean is None:
            return x
        else:
            return x, mean, std

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, 
                             hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, 
                             hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, 
                             hidden_size, kernel_size, padding=padding)

        self.out_act = nn.Tanh()
        self.sigmoid = nn.Sigmoid() 

    def forward(self, input_, prev_state=None):
        
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        device = input_.get_device()
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if input_.is_cuda:
                prev_state = torch.zeros(state_size).cuda(device)
            else :
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = self.sigmoid(self.update_gate(stacked_inputs))
        reset = self.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = self.out_act(self.out_gate(\
                     torch.cat([input_, prev_state * reset], dim=1)))
       # out_inputs = self.out_act(self.out_gate(torch.cat([input_,
                                         #prev_state], dim=1))) 
       #                                  prev_state * reset], dim=1)))

        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original
            Tensorflow version.
        """

        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
                      torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, 
                           self.weight * self.w_lrmul, 
                           self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)

        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


###########################################

class init_block(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim = 512):
        super().__init__()

        self.in_chan = in_channel
        self.out_chan = out_channel
        self.style_dim = style_dim

        self.init_const = ConstantInput(in_channel)
        self.init_noise = equal_lr(NoiseInjection(out_channel))
        self.adain = AdaptiveINstanceNorm(out_channel, style_dim)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise) :

        output = self.init_const(x)
        output = self.init_noise(output, noise) 
        output = self.adain(output, style)
        output = self.lrelu(output)       

        return output

#################################################################       
# ResBlock

class ResBlock(nn.Module):
    def __init__(self, enc_num,
                 kernel_size=3, stride=1, padding=1,
                  norm='spec', activation='relu'):
        super(ResBlock, self).__init__()

        self.conv1 = ConvBlock(enc_num, enc_num,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               norm=norm, activation=activation)
        self.conv2 = ConvBlock(enc_num, enc_num,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding,
                               norm=norm, activation=activation)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class Preact_ResBlock(nn.Module):
    def __init__(self, enc_num, 
                 kernel_size=3, stride=1, padding=1, 
                  norm='spec', activation='relu'):
        super(Preact_ResBlock, self).__init__()

        self.conv1 = Preact_ConvBlock(enc_num, enc_num, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      padding=padding, 
                                      norm=norm, activation=activation)
        self.conv2 = Preact_ConvBlock(enc_num, enc_num, 
                                      kernel_size=kernel_size, 
                                      stride=1, 
                                      padding=padding, 
                                      norm=norm, activation=activation)
         
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual
        
class Preact_ResBlock_avg_pool(nn.Module):
    def __init__(self, input_chan, output_chan,
                 kernel_size=3, stride = 1, padding=1, 
                 norm='spec', activation='relu'):

        super(Preact_ResBlock_avg_pool, self).__init__()
        self.pono = PONO()
        self.avgpool = nn.AvgPool2d(2)
        self.conv1 = Preact_ConvBlock(input_chan, output_chan, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      padding=padding,
                                      norm= norm, activation=activation)
        self.conv2 = Preact_ConvBlock(output_chan, output_chan,
                                      kernel_size,
                                      stride=1,
                                      padding= 1,
                                      norm=norm, activation=activation)

        self.residual = Preact_ConvBlock(input_chan, output_chan,
                                      kernel_size= kernel_size,
                                      stride = stride,
                                      padding = padding,
                                      norm = 'none', activation = 'none')
        self.norm = norm

    def forward(self, x):
        if self.norm == 'pono':
            out, mean, std = self.pono(x)
            residual = self.residual(out)
          
            out = self.conv1(out)
            out = self.conv2(out)
            out = self.avgpool(out+residual)
            mean = self.avgpool(mean)
            std = self.avgpool(std) 
            return out, mean, std

        else :
            residual = self.residual(x)
            out = self.conv1(x)
            out = self.conv2(x)
            out = self.avgpool(out+residual)
            return x + out
            

class Preact_ResBlock_upsample(nn.Module):
    def __init__(self, input_chan, output_chan, 
                 kernel_size=3, stride = 1, padding=1, 
                 norm='spec', activation='relu'):
        super(Preact_ResBlock_upsample, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode = 'nearest')        
        self.conv1 = Preact_ConvBlock(input_chan, output_chan, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      padding=padding, 
                                      norm=norm, activation=activation)
        self.conv2 = Preact_ConvBlock(output_chan, output_chan,
                                      kernel_size,
                                      stride=1,
                                      padding=1,
                                      norm=norm, activation=activation)

        self.residual = ConvBlock(input_chan, output_chan,
                                  kernel_size = kernel_size,
                                  stride = 1,
                                  padding = 1,
                                  norm = 'none', activation = 'none')

    def forward(self, x):
        x = self.up(x)
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out+residual
    
class Preact_DoubleBlock(nn.Module):
    def __init__(self, input_chan, output_chan,
                 kernel_size=3, stride = 1, padding=1,
                 norm='spec', activation='relu'):
        super(Preact_DoubleBlock, self).__init__()

        self.conv1 = Preact_ConvBlock(input_chan, output_chan,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      norm=norm, activation=activation)
        self.conv2 = Preact_ConvBlock(output_chan, output_chan,
                                      kernel_size = kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      norm=norm, activation=activation)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DoubleBlock(nn.Module):
    def __init__(self, input_chan, output_chan,
                first_kernel=3, second_kernel=3, stride = 1, padding=1,
                style_num = 3,
                first_norm='spec', second_norm = 'in',
                first_act='relu', second_act='relu'):
        super(DoubleBlock, self).__init__()
        if first_kernel ==3 :
            self.conv1 = ConvBlock(input_chan, output_chan,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                norm=first_norm, activation=first_act, style_num = style_num)
        elif first_kernel == 1:
            self.conv1 = ConvBlock(input_chan, output_chan,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm=first_norm, activation=first_act, style_num =style_num)

        self.conv2 = ConvBlock(output_chan, output_chan,
                                kernel_size = second_kernel,
                                stride=stride,
                                padding=padding,
                                norm=second_norm, activation=second_act, style_num = style_num)

    def forward(self,x, style_id=0):
        x = self.conv1(x, style_id)
        x = self.conv2(x, style_id)
        return x

class Double_GRU(nn.Module):
    def __init__(self, input_chan, output_chan,
                 first_kernel = 3, second_kernel=3, stride=1, padding=1,
                 norm='spec', 
                 first_act ='relu', second_act='relu'):
        super(Double_GRU, self).__init__()

        self.conv_gru = ConvGRUCell(input_chan=input_chan,
                                      hidden_chan= output_chan,
                                      kernel_size=first_kernel)

        self.conv_gru2 = ConvGRUCell(input_chan=input_chan,
                                      hidden_chan= output_chan,
                                      kernel_size=second_kernel)
        
        self.hidden_1 = None

    def forward(self, x):

        x = self.conv_gru(x, self.hidden_1) 
        x = self.conv_gru2(x,self.hidden_1)

        x = self.conv(x)
        return x

class Recurrent_unit(nn.Module):
    def __init__(self, input_chan, output_chan,
                 kernel_size = 3,
                 norm = 'in', act = 'relu'):
        super(Recurrent_unit, self).__init__()
        self.gru_block = ConvGRUCell(input_chan, output_chan,
                                 kernel_size= kernel_size)

        self.conv_block = DoubleBlock(output_chan, output_chan,
                            first_kernel= kernel_size, second_kernel=kernel_size,
                            stride = 1, padding =1,
                            first_norm = norm, second_norm = norm,
                            first_act = act, second_act = act)
    
                            
    def forward(self, in_x, hidden=None):
        hidden = self.gru_block(in_x, hidden)
        x = self.conv_block(hidden)
    
        return x, hidden
