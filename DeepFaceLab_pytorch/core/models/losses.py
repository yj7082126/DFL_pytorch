import torch
import torch.nn.functional as F
from core.imagelib import gaussian_blur, style_loss, total_variation_loss

def ganloss_step(src, result, discriminator, D_Loss, optim_dict):
    if optim_dict.masked_training:
        target_src_opt       = src["target"] * src["blur_mask"]
        prd_src_src_opt      = result["prd_src_src"] * src["blur_mask"]
        target_src_anti_opt  = src["target"] * (1.0 - src["blur_mask"])
        prd_src_src_anti_opt = result["prd_src_src"] * (1.0 - src["blur_mask"])
    else:
        target_src_opt  = src["target"]
        prd_src_src_opt = result["prd_src_src"]
        
    prd_src_src_d1, prd_src_src_d2 = discriminator(prd_src_src_opt.detach())
    tgt_src_d1, tgt_src_d2 = discriminator(target_src_opt)
    
    prd_src_src_d1_ones  = torch.ones_like(prd_src_src_d1)
    prd_src_src_d1_zeros = torch.zeros_like(prd_src_src_d1)
    prd_src_src_d2_ones  = torch.ones_like(prd_src_src_d2)
    prd_src_src_d2_zeros = torch.zeros_like(prd_src_src_d2)

    tgt_src_d1_ones  = torch.ones_like(tgt_src_d1)
    tgt_src_d2_ones  = torch.ones_like(tgt_src_d2)

    disc_loss_1 = D_Loss(tgt_src_d1_ones, tgt_src_d1) + D_Loss(prd_src_src_d1_zeros, prd_src_src_d1)
    disc_loss_2 = D_Loss(tgt_src_d2_ones, tgt_src_d2) + D_Loss(prd_src_src_d2_zeros, prd_src_src_d2)
    disc_loss = (disc_loss_1 + disc_loss_2) * 0.5

    prd_src_src_d1, prd_src_src_d2 = discriminator(prd_src_src_opt)

    g_loss_1 = D_Loss(prd_src_src_d1_ones, prd_src_src_d1)
    g_loss_2 = D_Loss(prd_src_src_d2_ones, prd_src_src_d2)
    g_loss = (g_loss_1 + g_loss_2)
    
    if optim_dict.masked_training:
        g_loss += total_variation_loss(result["prd_src_src"])
        g_loss += 0.02 * torch.square(prd_src_src_anti_opt-target_src_anti_opt).mean(axis=[0,1,2,3])
    
    return disc_loss, g_loss

def loss_step(src, dst, result, dssimloss, mseloss, optim_dict, radius, 
                discriminator=None, D_Loss=None):
    if optim_dict.masked_training:
        target_src_opt = src["target"] * src["blur_mask"]
        target_dst_opt = dst["target"] * dst["blur_mask"]
        prd_src_src_opt = result["prd_src_src"] * src["blur_mask"]
        prd_dst_dst_opt = result["prd_dst_dst"] * dst["blur_mask"]
    else:
        target_src_opt = src["target"]
        target_dst_opt = dst["target"]
        prd_src_src_opt = result["prd_src_src"]
        prd_dst_dst_opt = result["prd_dst_dst"]
        
    src_dssim_val1 = dssimloss(target_src_opt, prd_src_src_opt)
    src_mse_val    = mseloss(target_src_opt,  prd_src_src_opt)
    src_mask_val   = mseloss(src["mask"], result["prd_src_srcm"])
    src_lossval    = src_dssim_val1 + src_mse_val + src_mask_val

    dst_dssim_val1 = dssimloss(target_dst_opt, prd_dst_dst_opt)
    dst_mse_val    = mseloss(target_dst_opt,  prd_dst_dst_opt)
    dst_mask_val   = mseloss(dst["mask"], result["prd_dst_dstm"])
    dst_lossval    = dst_dssim_val1 + dst_mse_val + dst_mask_val   
    
    if optim_dict.eyes_mouth_prio:
        target_src_eye_opt = src["target"] * src["eyemask"]
        target_dst_eye_opt = dst["target"] * dst["eyemask"]
        prd_src_src_eye_opt = result["prd_src_src"] * src["eyemask"]
        prd_dst_dst_eye_opt = result["prd_dst_dst"] * dst["eyemask"]
        
        src_eye_val = F.l1_loss(prd_src_src_eye_opt, target_src_eye_opt) * 30
        dst_eye_val = F.l1_loss(prd_dst_dst_eye_opt, target_dst_eye_opt) * 30
        src_lossval += src_eye_val
        dst_lossval += dst_eye_val
        
    tot_lossval    = src_lossval + dst_lossval

    if optim_dict.true_style_pow > 0.0:
        dst_styleblur_mask   = gaussian_blur(dst["mask"], (radius // 32), use_cpu=False)
        target_dst_styleopt  = dst["target"] * dst_styleblur_mask
        prd_dst_dst_styleopt = result["prd_dst_dst"] * dst_styleblur_mask
        tot_stylelossval = style_loss(
            prd_dst_dst_styleopt, target_dst_styleopt, (radius // 16)
        ).sum()
        tot_stylelossval = optim_dict.true_style_pow * tot_stylelossval
        
        tot_lossval += tot_stylelossval
        
    if optim_dict.use_gan:
        disc_loss, g_loss = ganloss_step(src, result, discriminator, D_Loss, optim_dict)
        disc_loss = optim_dict.gan_pow * disc_loss
        g_loss = optim_dict.gan_pow * g_loss
        tot_lossval += g_loss

    losses = {
        "Total_Loss"    : tot_lossval.mean(),
        "Src_Loss"      : src_lossval.mean(), 
        "Dst_Loss"      : dst_lossval.mean()
    }
    if optim_dict.eyes_mouth_prio:
        losses["Src_Loss_Eyes"] = src_eye_val.mean()
        losses["Dst_Loss_Eyes"] = dst_eye_val.mean()
    if optim_dict.true_style_pow > 0.0:
        losses["Style_Loss"] = tot_stylelossval.mean()
    if optim_dict.use_gan:
        losses["GAN_Loss"] = g_loss
        losses["Disc_Loss"] = disc_loss
    
    images = {
        "Target SRC"      : target_src_opt, 
        "Target DST"      : target_dst_opt, 
        "Predict SRC"     : prd_src_src_opt, 
        "Predict DST"     : prd_dst_dst_opt, 
        "Predict SRC-DST" : result["prd_src_dst"]
    }
    return tot_lossval, losses, images