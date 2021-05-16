import sys
from pathlib import Path
import shutil
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.datasets import SampleFaceDataset, MergeDataset, UniformYawBatchSampler
from core.imagelib import DSSIM
from core.models import DF, LIAE, UNetPatchDiscriminator, loss_step
from core.options import read_yaml
from core.loglib import write_losses, write_images, save_weights, load_weights

#%%
def train(src_path, dst_path, model_path, config_path, gpu_idxs, 
            savedmodel_path=None):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    model_path = Path(model_path)
    
    finetune_start = False
    device = f"cuda:{gpu_idxs[0]}"
    parallel = len(gpu_idxs) > 1

    if savedmodel_path:
        savedmodel_path = Path(savedmodel_path)
        if not model_path.is_dir():
            shutil.copytree(savedmodel_path, model_path)
            finetune_start = True

    debug_path = model_path.joinpath("history")
    debug_path.mkdir(parents=True, exist_ok=True)
    backup_path = model_path.joinpath("autobackups")
    backup_path.mkdir(parents=True, exist_ok=True)

    config = read_yaml(config_path)
    data_dict  = config.data_params
    model_dict = read_yaml(model_path.joinpath("model_opt.yaml"))
    optim_dict = config.optim_params
    train_dict = config.train_params
    if parallel:
        train_dict.batch_size = train_dict.batch_size * len(gpu_idxs)

#%%
    src_dataset = SampleFaceDataset(src_path, 
                model_dict.resolution, model_dict.face_type, 
                data_dict.random_warp, data_dict.random_flip)
    dst_dataset = SampleFaceDataset(dst_path, 
                model_dict.resolution, model_dict.face_type, 
                data_dict.random_warp, data_dict.random_flip)
    mrg_dataset = MergeDataset(src_dataset, dst_dataset)

    if data_dict.uniform_yaw:
        src_yaw_dist = src_dataset.get_yaw_dist()
        src_sampler = UniformYawBatchSampler(src_yaw_dist, batch_size=train_dict.batch_size)
        train_loader = DataLoader(mrg_dataset, batch_sampler=src_sampler)
    else:
        train_loader = DataLoader(mrg_dataset, batch_size=train_dict.batch_size)

#%%
    if model_dict.model_type.startswith('df'):
        model = DF(model_dict.resolution, 
                model_dict.ae_dims, model_dict.e_dims, 
                model_dict.d_dims, model_dict.d_mask_dims, 
                likeness=model_dict.likeness, double_res=model_dict.double_res).to(device)
    else:
        model = LIAE(model_dict.resolution, 
                model_dict.ae_dims, model_dict.e_dims, 
                model_dict.d_dims, model_dict.d_mask_dims, 
                likeness=model_dict.likeness, double_res=model_dict.double_res).to(device)
        
    model, log_history = load_weights(model_path, model, finetune_start=finetune_start)
    start_iters = log_history["current_iters"]

    mseloss = nn.MSELoss()
    dssimloss = DSSIM(filter_size = int(model_dict.resolution // 11.6)).to(device)

    if optim_dict.use_gan:
        discriminator = UNetPatchDiscriminator(
            optim_dict.gan_patch_size, 3, optim_dict.gan_dims
        ).to(device)
        D_Loss = nn.BCEWithLogitsLoss()

    if parallel:
        model = nn.DataParallel(model, device_ids=gpu_idxs)
        dssimloss = nn.DataParallel(dssimloss, device_ids=gpu_idxs)
        if optim_dict.use_gan:
            discriminator = nn.DataParallel(discriminator, device_ids=gpu_idxs)

#%%
    model_opt = optim.Adam(model.parameters(), lr=optim_dict.learning_rate)

    if optim_dict.lr_schedule:
        scheduler = optim.lr_scheduler.StepLR(model_opt, 500, gamma=0.1)
        
    if optim_dict.use_gan:
        disc_opt = optim.Adam(discriminator.parameters(), lr=optim_dict.learning_rate)
        if optim_dict.lr_schedule:
            disc_scheduler = optim.lr_scheduler.StepLR(disc_opt, 500, gamma=0.1)

    writer = SummaryWriter(log_dir=str(model_path))
    curr_iters = 0

    pbar = tqdm(total=train_dict.target_iter)

    while True:
        if data_dict.uniform_yaw:
            src_yaw_dist = src_dataset.get_yaw_dist()
            src_sampler = UniformYawBatchSampler(src_yaw_dist, batch_size=train_dict.batch_size)
            train_loader = DataLoader(mrg_dataset, batch_sampler=src_sampler)

        for iter, data in enumerate(train_loader):
            curr_iters += train_dict.batch_size
            log_history["current_iter"] = start_iters+curr_iters
            
            for key, item in data.items():
                for key2, item2 in item.items():
                    if key2 != "filename":
                        data[key][key2] = item2.to(device)

            model_opt.zero_grad()
            if optim_dict.use_gan:
                disc_opt.zero_grad()

            result = model(data["src"]["warped"], data["dst"]["warped"])
            if optim_dict.use_gan:
                tot_lossval, losses, images = loss_step(data["src"], data["dst"], result,
                            dssimloss, mseloss, optim_dict, model_dict.resolution, 
                            discriminator, D_Loss)
            else:
                tot_lossval, losses, images = loss_step(data["src"], data["dst"], result,
                            dssimloss, mseloss, optim_dict, model_dict.resolution)                
            tot_lossval.sum().backward()
            
            model_opt.step()
            if optim_dict.lr_schedule:
                scheduler.step()
                
            if optim_dict.use_gan:
                losses["Disc_Loss"].backward()
                disc_opt.step()
                if optim_dict.lr_schedule:
                    disc_scheduler.step()

            pbar.update(train_dict.batch_size)
            pbar.set_postfix(src=f"{losses['Src_Loss']:.5f}", dst=f"{losses['Dst_Loss']:.5f}")

            if curr_iters % train_dict.debug_iter == 0:      
                write_losses(writer, losses, start_iters+curr_iters)

                log_history["src_loss_history"][curr_iters] = losses['Src_Loss'].item()
                log_history["dst_loss_history"][curr_iters] = losses['Dst_Loss'].item()

            if curr_iters % train_dict.preview_iter == 0:
                write_images(writer, images, debug_path, start_iters+curr_iters)

            if curr_iters % train_dict.save_iter == 0:
                save_weights(model_path, model.module if parallel else model, 
                            log_history=log_history, config=config)

            if curr_iters % train_dict.backup_iter == 0:
                save_weights(backup_path.joinpath(f"{curr_iters:05d}"), model.module if parallel else model, 
                            log_history=log_history, config=config)
                
            if curr_iters >= train_dict.target_iter:
                save_weights(model_path, model.module if parallel else model, 
                            log_history=log_history, config=config)
                save_weights(backup_path.joinpath(f"{curr_iters:05d}"), model.module if parallel else model, 
                                        log_history=log_history, config=config)
                sys.exit()


    pbar.close()