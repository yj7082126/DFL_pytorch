from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn

from core.models import blocks

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, n_downscales=4):
        super(Encoder, self).__init__()

        self.down1 = blocks.DownscaleBlock(in_ch, out_ch, n_downscales)

    def forward(self, x):
        out = self.down1(x)
        return torch.flatten(out, start_dim=1)

class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, lowest_dense_res, 
                 likeness=False):
        super(Inter, self).__init__()

        self.new_shape =  (-1, ae_out_ch, lowest_dense_res, lowest_dense_res)

        self.dense1 = nn.Linear(in_ch, ae_ch)
        self.dense2 = nn.Linear(ae_ch, ae_out_ch * (lowest_dense_res ** 2))
        self.upscale1 = blocks.Upscale(ae_out_ch, ae_out_ch)
        
        self.likeness = likeness
        if self.likeness:
            self.dense_norm = blocks.DenseNorm()

        self.out_res = lowest_dense_res * 2
        self.out_ch  = ae_out_ch

    def forward(self, inp):
        x = inp
        if self.likeness:
            x = self.dense_norm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.reshape(x, self.new_shape)
        x = self.upscale1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_ch, d_ch, d_mask_ch, 
                double_res=False, resolution=288):
        super(Decoder, self).__init__()
        
        self.upscale0 = blocks.Upscale(in_ch,  d_ch*8, kernel_size=3)
        self.upscale1 = blocks.Upscale(d_ch*8, d_ch*4, kernel_size=3)
        self.upscale2 = blocks.Upscale(d_ch*4, d_ch*2, kernel_size=3)
        
        self.res0 = blocks.ResidualBlock(d_ch*8, kernel_size=3)
        self.res1 = blocks.ResidualBlock(d_ch*4, kernel_size=3)
        self.res2 = blocks.ResidualBlock(d_ch*2, kernel_size=3)
        
        self.upscalem0 = blocks.Upscale(in_ch,       d_mask_ch*8, kernel_size=3)
        self.upscalem1 = blocks.Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
        self.upscalem2 = blocks.Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
        
        self.double_res = double_res
        if self.double_res:
            self.out_conv  = nn.Conv2d(d_ch*2, 3, 1, padding=0)
            self.out_conv1 = nn.Conv2d(d_ch*2, 3, 3, padding=1)
            self.out_conv2 = nn.Conv2d(d_ch*2, 3, 3, padding=1)
            self.out_conv3 = nn.Conv2d(d_ch*2, 3, 3, padding=1)
            self.upsample  = nn.Upsample(scale_factor=2)

            z0 = np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1,1,2,2)
            z0 = np.tile(z0, (resolution // 2, resolution // 2))
            self.register_buffer('z0', torch.from_numpy(z0))
            z1 = np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1,1,2,2)
            z1 = np.tile(z1, (resolution // 2, resolution // 2))
            self.register_buffer('z1', torch.from_numpy(z1))
            z2 = np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1,1,2,2)
            z2 = np.tile(z2, (resolution // 2, resolution // 2))
            self.register_buffer('z2', torch.from_numpy(z2))
            z3 = np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1,1,2,2)
            z3 = np.tile(z3, (resolution // 2, resolution // 2))
            self.register_buffer('z3', torch.from_numpy(z3))

            self.upscalem3 = blocks.Upscale(d_mask_ch*2, d_mask_ch, kernel_size=3)
            self.out_convm = nn.Conv2d(d_mask_ch, 1, 1, padding=0) 
        else:
            self.out_conv  = nn.Conv2d(d_ch*2,      3, 1, padding=0)
            self.out_convm = nn.Conv2d(d_mask_ch*2, 1, 1, padding=0) 

    def forward(self, z):
        x = self.res0(self.upscale0(z))
        x = self.res1(self.upscale1(x))
        x = self.res2(self.upscale2(x))

        if self.double_res:
            x0 = torch.sigmoid(self.out_conv(x))
            x1 = torch.sigmoid(self.out_conv1(x))
            x2 = torch.sigmoid(self.out_conv2(x))
            x3 = torch.sigmoid(self.out_conv3(x))
            x  = self.upsample(x0)*self.z0 + \
                 self.upsample(x1)*self.z1 + \
                 self.upsample(x2)*self.z2 + \
                 self.upsample(x3)*self.z3
        else:
            x = torch.sigmoid(self.out_conv(x))
        
        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        if self.double_res:
            m = self.upscalem3(m)
        m = torch.sigmoid(self.out_convm(m))

        return x, m

class DF(nn.Module):
    def __init__(self, resolution, ae_dims, e_dims, 
                 d_dims, d_mask_dims, 
                 likeness=False, double_res=False):
        super(DF, self).__init__()

        enc_total = (resolution ** 2) * e_dims // 32
        lowest_res = resolution // 32 if double_res else resolution // 16

        self.encoder     = Encoder(3, e_dims)
        self.inter       = Inter(enc_total, ae_dims, ae_dims, lowest_res, 
                            likeness=likeness)
        self.decoder_src = Decoder(ae_dims, d_dims, d_mask_dims,
                            double_res=double_res, resolution=resolution)
        self.decoder_dst = Decoder(ae_dims, d_dims, d_mask_dims,
                            double_res=double_res, resolution=resolution)

    def forward(self, warped_src, warped_dst):
        src_code = self.inter(self.encoder(warped_src))
        dst_code = self.inter(self.encoder(warped_dst))
        prd_src_src, prd_src_srcm = self.decoder_src(src_code)
        prd_dst_dst, prd_dst_dstm = self.decoder_dst(dst_code)
        prd_src_dst, prd_src_dstm = self.decoder_src(dst_code)

        return {
            "src_code"     : src_code,
            "dst_code"     : dst_code,
            "prd_src_src"  : prd_src_src, 
            "prd_src_srcm" : prd_src_srcm,
            "prd_dst_dst"  : prd_dst_dst, 
            "prd_dst_dstm" : prd_dst_dstm,
            "prd_src_dst"  : prd_src_dst, 
            "prd_src_dstm" : prd_src_dstm,            
        }

    def single_forward(self, warped_dst):
        dst_code = self.inter(self.encoder(warped_dst))
        _, prd_dst_dstm = self.decoder_dst(dst_code)
        prd_src_dst, prd_src_dstm = self.decoder_src(dst_code)

        return prd_src_dst, prd_src_dstm, prd_dst_dstm

class LIAE(nn.Module):
    def __init__(self, resolution, ae_dims, e_dims, 
                 d_dims, d_mask_dims, 
                 likeness=False, double_res=False):
        super(LIAE, self).__init__()

        enc_total = (resolution ** 2) * e_dims // 32
        lowest_res = resolution // 32 if double_res else resolution // 16

        self.encoder     = Encoder(3, e_dims)
        self.inter_AB    = Inter(enc_total, ae_dims, ae_dims*2, lowest_res, likeness=likeness)
        self.inter_B     = Inter(enc_total, ae_dims, ae_dims*2, lowest_res, likeness=likeness)
        self.decoder     = Decoder(ae_dims*4, d_dims, d_mask_dims,
                            double_res=double_res, resolution=resolution)

    def forward(self, warped_src, warped_dst):
        src_code = self.encoder(warped_src)
        src_interab_code = self.inter_AB(src_code)
        src_code = torch.cat([
            src_interab_code, src_interab_code
        ], axis=1)

        dst_code = self.encoder(warped_dst)
        dst_interb_code = self.inter_B(dst_code)
        dst_interab_code = self.inter_AB(dst_code)
        dst_code = torch.cat([
            dst_interb_code, dst_interab_code
        ], axis=1)
        
        src_dst_code = torch.cat([
            dst_interab_code, dst_interab_code
        ], axis=1)

        prd_src_src, prd_src_srcm = self.decoder(src_code)
        prd_dst_dst, prd_dst_dstm = self.decoder(dst_code)
        prd_src_dst, prd_src_dstm = self.decoder(src_dst_code)

        return {
            "src_code"     : src_code,
            "dst_code"     : dst_code,
            "prd_src_src"  : prd_src_src, 
            "prd_src_srcm" : prd_src_srcm,
            "prd_dst_dst"  : prd_dst_dst, 
            "prd_dst_dstm" : prd_dst_dstm,
            "prd_src_dst"  : prd_src_dst, 
            "prd_src_dstm" : prd_src_dstm,            
        }

    def single_forward(self, warped_dst):
        dst_code = self.encoder(warped_dst)
        dst_interb_code = self.inter_B(dst_code)
        dst_interab_code = self.inter_AB(dst_code)

        dst_code = torch.cat([
            dst_interb_code, dst_interab_code
        ], axis=1)
        src_dst_code = torch.cat([
            dst_interab_code, dst_interab_code
        ], axis=1)
        _, prd_dst_dstm = self.decoder(dst_code)
        prd_src_dst, prd_src_dstm = self.decoder(src_dst_code)

        return prd_src_dst, prd_src_dstm, prd_dst_dstm

class LIAE_multi(nn.Module):
    def __init__(self, resolution, ae_dims, e_dims, 
                 d_dims, d_mask_dims, 
                 likeness=False, double_res=False):
        super(LIAE_multi, self).__init__()

        enc_total = (resolution ** 2) * e_dims // 32
        lowest_res = resolution // 32 if double_res else resolution // 16

        self.encoder_id   = Encoder(3, e_dims)
        self.encoder_attr = Encoder(3, e_dims)
        self.inter_id     = Inter(enc_total, ae_dims, ae_dims*2, lowest_res, likeness=likeness)
        self.inter_attr   = Inter(enc_total, ae_dims, ae_dims*2, lowest_res, likeness=likeness)
        self.decoder      = Decoder(ae_dims*4, d_dims, d_mask_dims,
                            double_res=double_res, resolution=resolution)

    def forward(self, warped_src, warped_dst):
        src_id_code = self.inter_id(self.encoder_id(warped_src))
        dst_id_code = self.inter_id(self.encoder_id(warped_dst))
        src_attr_code = self.inter_attr(self.encoder_attr(warped_src))
        dst_attr_code = self.inter_attr(self.encoder_attr(warped_dst))
        
        src_code = torch.cat([src_id_code, src_attr_code], axis=1)
        dst_code = torch.cat([dst_id_code, dst_attr_code], axis=1)
        src_dst_code = torch.cat([src_id_code, dst_attr_code], axis=1)

        prd_src_src, prd_src_srcm = self.decoder(src_code)
        prd_dst_dst, prd_dst_dstm = self.decoder(dst_code)
        prd_src_dst, prd_src_dstm = self.decoder(src_dst_code)
        
        res_id_code = self.inter_id(self.encoder_id(prd_src_dst))
        res_attr_code = self.inter_attr(self.encoder_attr(prd_src_dst))
        
        src_res_code = torch.cat([src_id_code, res_attr_code], axis=1)
        res_dst_code = torch.cat([res_id_code, dst_attr_code], axis=1)
        
        prd_src_res, prd_src_resm = self.decoder(src_res_code)
        prd_res_dst, prd_res_dstm = self.decoder(res_dst_code)

        return {
            "src_code"     : src_code,
            "dst_code"     : dst_code,
            "res_id_code" : res_id_code,
            "res_attr_code" : res_attr_code,
            "prd_src_src"  : prd_src_src, 
            "prd_src_srcm" : prd_src_srcm,
            "prd_dst_dst"  : prd_dst_dst, 
            "prd_dst_dstm" : prd_dst_dstm,
            "prd_src_dst"  : prd_src_dst, 
            "prd_src_dstm" : prd_src_dstm, 
            "prd_src_res"  : prd_src_res,
            "prd_src_resm" : prd_src_resm,
            "prd_res_dst"  : prd_res_dst,
            "prd_res_dstm" : prd_res_dstm
        }

    def single_forward(self, warped_dst):
        dst_code = self.encoder(warped_dst)
        dst_interb_code = self.inter_B(dst_code)
        dst_interab_code = self.inter_AB(dst_code)

        dst_code = torch.cat([
            dst_interb_code, dst_interab_code
        ], axis=1)
        src_dst_code = torch.cat([
            dst_interab_code, dst_interab_code
        ], axis=1)
        _, prd_dst_dstm = self.decoder(dst_code)
        prd_src_dst, prd_src_dstm = self.decoder(src_dst_code)

        return prd_src_dst, prd_src_dstm, prd_dst_dstm

class UNetPatchDiscriminator(nn.Module):
    def __init__(self, patch_size, in_ch, base_ch = 16):
        super(UNetPatchDiscriminator, self).__init__()
        
        layers = blocks.find_archi(patch_size)
        level_chs = { 
            i-1:v for i,v in enumerate([
                min( base_ch * (2**i), 512 ) for i in range(len(layers)+1)
            ])
        }
        
        self.convs = nn.ModuleList([])
        self.res1 = nn.ModuleList([])
        self.res2 = nn.ModuleList([])
        self.upconvs = nn.ModuleList([])
        self.upres1 = nn.ModuleList([])
        self.upres2 = nn.ModuleList([])
        prev_ch = in_ch
        
        self.in_conv = nn.Conv2d( in_ch, level_chs[-1], 1)

        for i, (kernel_size, strides) in enumerate(layers):
            self.convs.append( 
                blocks.Conv2d_SAME(level_chs[i-1], level_chs[i], kernel_size, stride=strides) 
            )

            self.res1.append(blocks.ResidualBlock(level_chs[i]))
            self.res2.append(blocks.ResidualBlock(level_chs[i]))
            
            self.upconvs.insert(0, 
                blocks.Conv2dTranspose_SAME( 
                    level_chs[i]*(2 if i != len(layers)-1 else 1), 
                    level_chs[i-1], 
                    kernel_size, stride=strides) )

            self.upres1.insert(0, blocks.ResidualBlock(level_chs[i-1]*2))
            self.upres2.insert(0, blocks.ResidualBlock(level_chs[i-1]*2))
            
        self.out_conv = nn.Conv2d(level_chs[-1]*2, 1, 1)

        self.center_out  =  nn.Conv2d(level_chs[len(layers)-1], 1, 1)
        self.center_conv =  nn.Conv2d(level_chs[len(layers)-1], level_chs[len(layers)-1], 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, x):
        x = self.leaky_relu(self.in_conv(x))

        encs = []
        for conv, res1,res2 in zip(self.convs, self.res1, self.res2):
            encs.insert(0, x)
            x = self.leaky_relu(conv(x))
            x = res2(res1(x))
            
        center_out, x = self.center_out(x), self.leaky_relu(self.center_conv(x))

        for i, (upconv, enc, upres1, upres2 ) in enumerate(zip(self.upconvs, encs, self.upres1, self.upres2)):
            x = self.leaky_relu(upconv(x))
            x = torch.cat([enc, x], axis=1)
            x = upres2(upres1(x))

        return self.sigmoid(center_out), self.sigmoid(self.out_conv(x))