import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(x, mu, sigma):
    num = -(float(x) - float(mu)) ** 2
    den = (2 * (sigma **2))
    res = np.exp(num / den)
    res = res * (1. / (2 * math.pi * (sigma ** 2)))
    return res

def make_kernel(sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = max(3, int(2 * 2 * sigma + 1))
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel = np_kernel / np.sum(np_kernel)
    return kernel, kernel_size

def gaussian_blur_params(sigma, channel=1, kernel_size=None):
    gauss_kernel, kernel_size = make_kernel(sigma, kernel_size)
    gauss_kernel = torch.from_numpy(gauss_kernel).view(1, 1, kernel_size, kernel_size)
    gauss_kernel = gauss_kernel.repeat(channel, 1, 1, 1)
    return gauss_kernel, kernel_size

def gaussian_blur(x, radius, use_clamp=True, 
        device=None, use_cpu=True):
    channel = x.shape[1]
    gauss_kernel, kernel_size = gaussian_blur_params(radius, channel=channel)

    if use_cpu == False:
        if device:
            gauss_kernel = gauss_kernel.to(device)
        else:
            gauss_kernel = gauss_kernel.to(x.get_device())

    x_blur = F.conv2d(x, gauss_kernel, padding = kernel_size//2, groups = channel)
    if use_clamp:
        x_blur = torch.clamp(x_blur, 0.0, 0.5) * 2
    return x_blur

def style_loss(target, style, gaussian_blur_radius=0.0):
    if gaussian_blur_radius > 0.0:
        target = gaussian_blur(target, gaussian_blur_radius, use_clamp=False, use_cpu=False)
        style  = gaussian_blur(style, gaussian_blur_radius, use_clamp=False, use_cpu=False)
    c_std, c_mean = torch.std_mean(target, axis=[2,3], keepdims=True)
    s_std, s_mean = torch.std_mean(style, axis=[2,3], keepdims=True)
    mean_loss = torch.square(c_mean-s_mean).sum(axis=[1,2,3])
    std_loss = torch.square(c_std-s_std).sum(axis=[1,2,3])
    return (mean_loss + std_loss) / 3

def total_variation_loss(img, weight=1.0):
    denom = img.shape[0]*img.shape[1]*img.shape[2]*img.shape[3]
    tv_h = torch.square(img[:,:,1:,:] - img[:,:,:-1,:]).sum(axis=[0,1,2,3])
    tv_w = torch.square(img[:,:,:,1:] - img[:,:,:,:-1]).sum(axis=[0,1,2,3])
    return weight*(tv_h+tv_w) / denom
    
class DSSIM(nn.Module):
    def __init__(self, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, 
                size_average=True):
        super(DSSIM, self).__init__()

        self.gauss_kernel, self.kernel_size = gaussian_blur_params(
            filter_sigma, channel=3, kernel_size=filter_size)
        self.gauss_kernel = nn.Parameter(self.gauss_kernel)
        #self.gauss_kernel = self.gauss_kernel.to(device)
        self.C1 = k1**2
        self.C2 = k2**2
        self.size_average = size_average
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image1, image2):
        mu1 = F.conv2d(image1, self.gauss_kernel, padding = self.kernel_size//2, groups = 3)
        mu2 = F.conv2d(image2, self.gauss_kernel, padding = self.kernel_size//2, groups = 3)
        mu1_sq  = mu1.pow(2)
        mu2_sq  = mu2.pow(2)
        mu1_mu2 = mu1*mu2        
        
        sig1_sq = F.conv2d(image1*image1, self.gauss_kernel, 
            padding = self.kernel_size//2, groups = 3) - mu1_sq
        sig2_sq = F.conv2d(image2*image2, self.gauss_kernel, 
            padding = self.kernel_size//2, groups = 3) - mu2_sq
        sig1_sig2 = F.conv2d(image1*image2, self.gauss_kernel,
            padding = self.kernel_size//2, groups = 3) - mu1_mu2
        
        ssim_num = (2*mu1_mu2+self.C1)*(2*sig1_sig2+self.C2)
        ssim_den = (mu1_sq+mu2_sq+self.C1)*(sig1_sq+sig2_sq+self.C2)
        ssim_map = ssim_num / ssim_den
        dssim_map = (1.0 - ssim_map) / 2.0        

        if self.size_average:
            return dssim_map.mean()
        else:
            return dssim_map.mean(1).mean(1).mean(1)        

# def dssim(image1, image2, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, 
#             size_average=True):
#     channel = image1.shape[1]
#     gauss_kernel, kernel_size = gaussian_blur_params(filter_sigma, channel=channel, 
#                                                 kernel_size=filter_size)
#     gauss_kernel = gauss_kernel.cuda(image1.get_device())
    
#     mu1 = F.conv2d(image1, gauss_kernel, padding = kernel_size//2, groups = channel)
#     mu2 = F.conv2d(image2, gauss_kernel, padding = kernel_size//2, groups = channel)
#     mu1_sq  = mu1.pow(2)
#     mu2_sq  = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sig1_sq = F.conv2d(image1*image1, gauss_kernel, padding = kernel_size//2, groups = channel) - mu1_sq
#     sig2_sq = F.conv2d(image2*image2, gauss_kernel, padding = kernel_size//2, groups = channel) - mu2_sq
#     sig1_sig2 = F.conv2d(image1*image2, gauss_kernel, padding = kernel_size//2, groups = channel) - mu1_mu2

#     C1 = k1**2
#     C2 = k2**2

#     ssim_num = (2*mu1_mu2+C1)*(2*sig1_sig2+C2)
#     ssim_den = (mu1_sq+mu2_sq+C1)*(sig1_sq+sig2_sq+C2)
#     ssim_map = ssim_num / ssim_den
#     dssim_map = (1.0 - ssim_map) / 2.0

#     if size_average:
#         return dssim_map.mean()
#     else:
#         return dssim_map.mean(1).mean(1).mean(1)