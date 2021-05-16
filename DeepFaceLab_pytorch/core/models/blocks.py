import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

def get_same_padding(kernel_size, transpose=False):
    if transpose:
        pad = (kernel_size * 2)
    else:
        pad = max((kernel_size // 2), 0)
    return (pad, pad, pad, pad)
    
def depth_to_space(x, factor=2):
    channel = x.shape[1]
    subchannel = channel // (factor**2)
    indices = np.concatenate([np.arange(i, channel, subchannel) for i in range(subchannel)])
    x = x[:,indices,:,:]
    x = F.pixel_shuffle(x, factor)
    return x

class Conv2d_SAME(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1,
                    custom_weight=None):
        super(Conv2d_SAME, self).__init__()
        
        self.pad_tuple = get_same_padding(kernel_size)
        if custom_weight is None:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 
                                kernel_size=kernel_size, stride=stride)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 
                                kernel_size=kernel_size, groups=in_ch, 
                                bias=False)
            self.conv1.weight.data = custom_weight
            self.conv1.weight.requires_grad = False
    
    def forward(self, x):
        out = F.pad(x, self.pad_tuple, "constant")
        out = self.conv1(out)
        return out


class Conv2dTranspose_SAME(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1,
                    custom_weight=None):
        super(Conv2dTranspose_SAME, self).__init__()
        
        #self.pad_tuple = get_same_padding(kernel_size, transpose=True)
        if custom_weight is None:
            self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 
                                kernel_size=kernel_size, stride=stride, 
                                padding=1, output_padding=stride-1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 
                                kernel_size=kernel_size, groups=in_ch, 
                                bias=False)
            self.conv1.weight.data = custom_weight
            self.conv1.weight.requires_grad = False
    
    def forward(self, x):
        #out = F.pad(x, self.pad_tuple, "constant")
        out = x
        out = self.conv1(out)
        return out

class Downscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=2, act_slope=0.1):
        super(Downscale, self).__init__()
        
        #self.conv1 = Conv2d_SAME(in_ch, out_ch, kernel_size, stride=stride)
        self.conv1 =  nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=max(kernel_size // 2, 0))
        self.act = nn.LeakyReLU(act_slope)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        return out
    
class Upscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, act_slope=0.1, mult_fact=2):
        super(Upscale, self).__init__()

        self.mult_fact = mult_fact
        #self.conv1 = Conv2d_SAME(in_ch, out_ch * (mult_fact ** 2), kernel_size, stride=stride)
        self.conv1 = nn.Conv2d(in_ch, out_ch * (mult_fact ** 2), kernel_size, stride=stride, padding=max(kernel_size // 2, 0))
        self.act = nn.LeakyReLU(act_slope)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = depth_to_space(out, self.mult_fact)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, stride=1, act_slope=0.2):
        super(ResidualBlock, self).__init__()
        
        #self.conv1 = Conv2d_SAME(ch, ch, kernel_size, stride=stride)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size, stride=stride, padding=max(kernel_size // 2, 0))
        #self.conv2 = Conv2d_SAME(ch, ch, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size, stride=stride, padding=max(kernel_size // 2, 0))
        self.act   = nn.LeakyReLU(act_slope)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(x + out)
        return out

class DownscaleBlock(nn.Module):
    def __init__(self, in_ch, ch, n_downscales, kernel_size=5, stride=2, 
                 act_slope=0.1, max_mul=8):
        super(DownscaleBlock, self).__init__()

        downs = []
        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * (min(2 ** i, max_mul))
            downs.append(
                Downscale(last_ch, cur_ch, kernel_size=kernel_size, 
                          stride=stride, act_slope=act_slope)
            )
            last_ch = cur_ch
        self.downs = nn.Sequential(*downs)

    def forward(self, x):
        out = self.downs(x)
        return out

class DenseNorm():
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x):
        out = torch.mean(torch.square(x), axis=-1).unsqueeze(-1)
        out = torch.rsqrt(out + self.eps)
        return x * out

def calc_receptive_field_size(layers):
    rf = 0
    ts = 1
    for i, (k, s) in enumerate(layers):
        if i == 0:
            rf = k
        else:
            rf += (k-1)*ts
        ts *= s
    return rf

def find_archi(target_patch_size, max_layers=9):
    s = {}
    for layers_count in range(1,max_layers+1):
        val = 1 << (layers_count-1)
        while True:
            val -= 1

            layers = []
            sum_st = 0
            layers.append ( [3, 2])
            sum_st += 2
            for i in range(layers_count-1):
                st = 1 + (1 if val & (1 << i) !=0 else 0 )
                layers.append ( [3, st ])
                sum_st += st                

            rf = calc_receptive_field_size(layers)

            s_rf = s.get(rf, None)
            if s_rf is None:
                s[rf] = (layers_count, sum_st, layers)
            else:
                if layers_count < s_rf[0] or \
                ( layers_count == s_rf[0] and sum_st > s_rf[1] ):
                    s[rf] = (layers_count, sum_st, layers)

            if val == 0:
                break

    x = sorted(list(s.keys()))
    q=x[np.abs(np.array(x)-target_patch_size).argmin()]
    return s[q][2]
