import numpy as np
import cv2

from core.imagelib import *
from facelib.LandmarksProcessor import *

def read_image(filepath):
    img = cv2.imread(str(filepath))
    img = (img / 255.).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img    

def get_full_face_mask(img, landmarks, eyebrows_expand_mod=1.0,
                       xseg_mask=None):   
    h, w, c = img.shape
    
    if xseg_mask is not None:           
        if xseg_mask.shape[0] != h or xseg_mask.shape[1] != w:
            xseg_mask = cv2.resize(xseg_mask, (w,h), 
                                   interpolation=cv2.INTER_CUBIC)                    
        mask = normalize_channels(xseg_mask, 1)
    else:
        mask = get_image_hull_mask (
            (h, w, c), landmarks, 
            eyebrows_expand_mod=eyebrows_expand_mod 
        )
    mask = np.clip(mask, 0, 1)
    return mask

def get_eyes_mouth_mask(img, landmarks, full_mask=None):
    h, w, c = img.shape

    eyes_mask  = get_image_eye_mask ((h,w,c), landmarks)
    mouth_mask = get_image_mouth_mask((h,w,c), landmarks)
    mask = np.clip(eyes_mask + mouth_mask, 0.0, 1.0)
    
    if full_mask is not None:
        full_mask[full_mask != 0.0] = 1.0
        mask = mask * full_mask
    return mask
    
def resize_img(img, face_type, landmarks, 
               type="face", target_face_type="whole_face",
               resolution=288):
    w = img.shape[1]
    if type == "face":
        borderMode = cv2.BORDER_REPLICATE
    else:
        borderMode = cv2.BORDER_CONSTANT

    if face_type != target_face_type:
        mat = get_transform_mat(landmarks, resolution, target_face_type)
        img = cv2.warpAffine(img, mat, 
                             (resolution, resolution), 
                             borderMode=borderMode, 
                             flags=cv2.INTER_CUBIC)
    elif w != resolution:
        img = cv2.resize(img,(resolution, resolution), 
                             interpolation=cv2.INTER_CUBIC)
    return img