from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from core.datasets.dataset_utils import *
from core.imagelib import gaussian_blur

from DFLIMG import *

class SampleFaceDataset(Dataset):
    
    def __init__(self, samples_path,
                resolution=288, face_type="whole_face",
                random_warp=False, random_flip=True, ext="jpg"):
        self.samples_path = Path(samples_path)
        self.resolution = resolution
        self.radius = max(1, resolution // 32)
        self.face_type = face_type
        self.random_warp = random_warp
        self.random_flip = random_flip
        
        self.image_paths = sorted(list(self.samples_path.glob(f"*.{ext}")))
        self.get_samples()

    def get_samples(self):
        self.filenames = []
        self.samples = {}
        for idx, filename in enumerate(tqdm(self.image_paths)):
            dflimg = DFLIMG.load(Path(filename))
            self.filenames.append(Path(filename).name)
            self.samples[idx] = dflimg
            
    def get_params(self):
        seed = np.random.randint(0x80000000)
        warp_rnd_state = np.random.RandomState (seed-1)            
        params = imagelib.gen_warp_params(
            self.resolution, self.random_flip,
            rnd_state=warp_rnd_state)
        return params

    def get_yaw_dist(self, grads=128):
        grads_space = np.linspace (-1.2, 1.2, grads)

        samples_pyr = [ ]
        for idx, sample in self.samples.items():
            landmarks = sample.get_landmarks()
            size = sample.get_shape()[1]
            pyr = estimate_pitch_yaw_roll(landmarks, size=size)
            samples_pyr.append((idx, pyr))
            
        yaws_sample_dict = {}

        for g in range(grads):
            yaw = grads_space[g]
            next_yaw = grads_space[g+1] if g < grads-1 else yaw

            yaw_samples = []
            for idx, pyr in samples_pyr:
                s_yaw = -pyr[1]
                if g == 0:
                    if s_yaw < next_yaw:
                        yaw_samples += [ idx ]
                elif g < grads-1:
                    if s_yaw >= yaw and s_yaw < next_yaw:
                        yaw_samples += [ idx ]
                else:
                    if s_yaw >= yaw:
                        yaw_samples += [ idx ]
            yaws_sample_dict[yaw] = yaw_samples
        return yaws_sample_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        sample = self.samples[idx]
        sample_face_type = sample.get_face_type()
        sample_landmarks = sample.get_landmarks()
        sample_xseg_mask = sample.get_xseg_mask()  
        eyebrows_expand_mod = sample.get_eyebrows_expand_mod()
        eyebrows_expand_mod = eyebrows_expand_mod if eyebrows_expand_mod is not None else 1.0
        
        params = self.get_params()
        img = read_image(self.samples_path.joinpath(filename))
        mask_full = get_full_face_mask(img, sample_landmarks, 
                    eyebrows_expand_mod, sample_xseg_mask)
        mask_eyes = get_eyes_mouth_mask(img, sample_landmarks, 
                    mask_full)
        
        img       = resize_img(img, sample_face_type, sample_landmarks, 
                              target_face_type=self.face_type, 
                              resolution=self.resolution, type="face")
        mask_full = resize_img(mask_full, sample_face_type, sample_landmarks, 
                              target_face_type=self.face_type, 
                              resolution=self.resolution, type="mask")
        mask_eyes = resize_img(mask_eyes, sample_face_type, sample_landmarks, 
                              target_face_type=self.face_type, 
                              resolution=self.resolution, type="mask")
        
        warped_img  = imagelib.warp_by_params (params, 
            img, self.random_warp, True, True, True)
        warped_img = np.clip(warped_img.astype(np.float32), 0, 1)
        
        img = imagelib.warp_by_params(params, 
            img, False, True, True, True)
        img = np.clip(img.astype(np.float32), 0, 1)
        
        mask_full = imagelib.warp_by_params(params, 
            mask_full, False, True, True, False, 
            cv2_inter=cv2.INTER_LINEAR)
        mask_eyes = imagelib.warp_by_params(params, 
            mask_eyes, False, True, True, False, 
            cv2_inter=cv2.INTER_LINEAR)        
        mask_eyes = mask_eyes / mask_eyes.max() if mask_eyes.max() != 0.0 else mask_eyes
        
        warped_img = torch.from_numpy(np.transpose(warped_img, (2,0,1) ))
        img        = torch.from_numpy(np.transpose(img, (2,0,1) ))
        mask_full  = torch.from_numpy(np.transpose(mask_full.copy(), (2,0,1) ))
        mask_eyes  = torch.from_numpy(np.transpose(mask_eyes.copy(), (2,0,1) ))
        mask_blur  = gaussian_blur(mask_full.unsqueeze(0), self.radius)[0]

        return { 
            "filename" : filename,
            "warped" : warped_img, 
            "target" : img,
            "mask" : mask_full, 
            "eyemask" : mask_eyes,
            "blur_mask" : mask_blur
        }        

class MergeDataset(Dataset):
    def __init__(self, src_dataset, dst_dataset):
        self.src_dataset = src_dataset
        self.dst_dataset = dst_dataset
        
        self.len_src = len(self.src_dataset)
        self.len_dst = len(self.dst_dataset)
        self.max_len = max(self.len_src, self.len_dst)
        self.lrg_set = "src" if self.len_src >= self.len_dst else "dst"
        
    def __len__(self):
        return self.max_len
    
    def __getitem__(self, idx):
        if self.lrg_set == "src":
            src_el = self.src_dataset[idx]
            dst_el = self.dst_dataset[idx % self.len_dst]
        else:
            src_el = self.src_dataset[idx % self.len_src]
            dst_el = self.dst_dataset[idx]
        return {"src" : src_el, "dst" : dst_el}