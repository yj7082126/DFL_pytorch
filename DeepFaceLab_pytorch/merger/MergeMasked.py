import numpy as np
import cv2

from core import imagelib
from facelib import FaceType, LandmarksProcessor

def get_size_and_mat(predictor_input_shape, img_face_landmarks, 
        output_face_scale, face_type, use_sr=False):
    if type(face_type) == str:
        face_type = FaceType.fromString(face_type)
    input_size = predictor_input_shape[0]
    mask_subres_size = input_size * 4
    output_size = mask_subres_size if use_sr else input_size

    output_face_scale = 1.0 + 0.01 * output_face_scale
    face_mat        = LandmarksProcessor.get_transform_mat (
        img_face_landmarks, output_size, face_type=face_type, scale=1.0
    )
    face_output_mat = LandmarksProcessor.get_transform_mat (
        img_face_landmarks, output_size, face_type=face_type, scale=output_face_scale
    )
    face_mask_output_mat = LandmarksProcessor.get_transform_mat (
        img_face_landmarks, mask_subres_size, face_type=face_type, scale=output_face_scale
    )

    size_dict = {
        "input"  : input_size, 
        "mask"   : mask_subres_size,
        "output" : output_size
    }
    mat_dict = {
        "face"        : face_mat,
        "face_output" : face_output_mat,
        "face_mask"   : face_mask_output_mat
    }
    return size_dict, mat_dict

def warp_and_clip(img, mat, size, use_clip=True, inverse=False, remove_noise=False):
    if type(size) != tuple:
        size = (size, size)

    if inverse:
        dst = cv2.warpAffine(img, mat, size, np.empty(size), 
                             flags = cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    else:
        dst = cv2.warpAffine(img, mat, size, flags = cv2.INTER_CUBIC)

    if len(dst.shape) == 2:
        dst = dst[...,None]

    if use_clip:
        dst = np.clip(dst, 0.0, 1.0)

    if remove_noise:
        dst [ dst < (1.0/255.0) ] = 0.0 

    return dst

def get_prediction(predictor_func, dst_face_bgr, dst_face_mask_a_0, input_size, output_size, 
                face_type=0.0, use_model=True, filepath=None):
                
    if use_model:
        predictor_input_bgr      = cv2.resize(dst_face_bgr, (input_size,input_size) )
        predicted = predictor_func (predictor_input_bgr)
        prd_face_bgr          = np.clip (predicted[0], 0, 1.0)
        prd_face_mask_a_0     = np.clip (predicted[1], 0, 1.0)
        prd_face_dst_mask_a_0 = np.clip (predicted[2], 0, 1.0)
    else:
        prd_face_bgr, prd_landmarks = predictor_func (filepath)
        if prd_face_bgr is None:
            prd_face_bgr = dst_face_bgr
            prd_face_mask_a_0 = dst_face_mask_a_0.copy()
        else:
            prd_face_mask = LandmarksProcessor.get_image_hull_mask (prd_face_bgr.shape, prd_landmarks)
            prd_face_mat  = LandmarksProcessor.get_transform_mat(prd_landmarks, output_size, face_type=face_type)
            prd_face_mask_a_0 = cv2.warpAffine( prd_face_mask, prd_face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
            prd_face_mask_a_0 = np.clip(prd_face_mask_a_0, 0, 1.0)
        prd_face_dst_mask_a_0 = dst_face_mask_a_0.copy()
        
    return prd_face_bgr, prd_face_mask_a_0, prd_face_dst_mask_a_0

def get_normfacemask(dst_face_mask_a_0, prd_face_mask_a_0, prd_face_dst_mask_a_0, 
                 mask_mode=1, output_size=256):
    if mask_mode == 1: #dst
        wrk_face_mask_a_0 = cv2.resize (dst_face_mask_a_0, (output_size,output_size), interpolation=cv2.INTER_CUBIC)
    else:
        if mask_mode == 2: #learned-prd
            wrk_face_mask_a_0 = prd_face_mask_a_0
        elif mask_mode == 3: #learned-dst
            wrk_face_mask_a_0 = prd_face_dst_mask_a_0
        elif mask_mode == 4: #learned-prd*learned-dst
            wrk_face_mask_a_0 = prd_face_mask_a_0*prd_face_dst_mask_a_0
        else: #learned-prd+learned-dst
            wrk_face_mask_a_0 = np.clip( prd_face_mask_a_0+prd_face_dst_mask_a_0, 0, 1)

    wrk_face_mask_a_0[ wrk_face_mask_a_0 < (1.0/255.0) ] = 0.0 # get rid of noise
    return wrk_face_mask_a_0

def get_erodeblurmask(wrk_face_mask_a_0, ero=20, blur=20, input_size=256):
    wrk_face_mask_a_0 = np.pad (wrk_face_mask_a_0, input_size)

    if ero > 0:
        wrk_face_mask_a_0 = cv2.erode(wrk_face_mask_a_0, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), 
            iterations = 1 
        )
    elif ero < 0:
        wrk_face_mask_a_0 = cv2.dilate(wrk_face_mask_a_0, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), 
            iterations = 1 
        )

    clip_size = input_size + blur // 2

    wrk_face_mask_a_0[:clip_size,:] = 0
    wrk_face_mask_a_0[-clip_size:,:] = 0
    wrk_face_mask_a_0[:,:clip_size] = 0
    wrk_face_mask_a_0[:,-clip_size:] = 0

    if blur > 0:
        blur = blur + (1-blur % 2)
        wrk_face_mask_a_0 = cv2.GaussianBlur(wrk_face_mask_a_0, (blur, blur) , 0)

    wrk_face_mask_a_0 = wrk_face_mask_a_0[input_size:-input_size,input_size:-input_size]
    wrk_face_mask_a_0 = np.clip(wrk_face_mask_a_0, 0, 1)
    return wrk_face_mask_a_0

def get_colortransfer(prd_face_bgr, dst_face_bgr, wrk_face_mask_area_a, color_transfer_mode=1):
    if color_transfer_mode == 1: #rct
        prd_face_bgr = imagelib.reinhard_color_transfer ( np.clip( prd_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8),
                                                            np.clip( dst_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8), )

        prd_face_bgr = np.clip( prd_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)
    elif color_transfer_mode == 2: #lct
        prd_face_bgr = imagelib.linear_color_transfer (prd_face_bgr, dst_face_bgr)
    elif color_transfer_mode == 3: #mkl
        prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr, dst_face_bgr)
    elif color_transfer_mode == 4: #mkl-m
        prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
    elif color_transfer_mode == 5: #idt
        prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr, dst_face_bgr)
    elif color_transfer_mode == 6: #idt-m
        prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
    elif color_transfer_mode == 7: #sot-m
        prd_face_bgr = imagelib.color_transfer_sot (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a, steps=10, batch_size=30)
        prd_face_bgr = np.clip (prd_face_bgr, 0.0, 1.0)
    elif color_transfer_mode == 8: #mix-m
        prd_face_bgr = imagelib.color_transfer_mix (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
    elif color_transfer_mode == 9: #disney
        prd_face_bgr = imagelib.disney_color_transfer (prd_face_bgr, dst_face_bgr, wrk_face_mask_area_a)
    return prd_face_bgr

def if_maskminsize(img_face_mask_a, min_val=0.1):
    maxregion = np.argwhere( img_face_mask_a >= min_val )
    if maxregion.size != 0:
        miny,minx = maxregion.min(axis=0)[:2]
        maxy,maxx = maxregion.max(axis=0)[:2]
        lenx = maxx - minx
        leny = maxy - miny
        if min(lenx,leny) >= 4:
            return True
        else:
            return False
    else:
        return False

def get_histmatch(prd_face_bgr, dst_face_bgr, wrk_face_mask_area_a, 
                    masked_hist_match=True, hist_match_threshold=239):
    hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

    if masked_hist_match:
        hist_mask_a *= wrk_face_mask_area_a

    white =  (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

    hist_match_1 = prd_face_bgr*hist_mask_a + white
    hist_match_1[ hist_match_1 > 1.0 ] = 1.0
    hist_match_2 = dst_face_bgr*hist_mask_a + white
    hist_match_2[ hist_match_1 > 1.0 ] = 1.0

    prd_face_bgr = imagelib.color_hist_match(hist_match_1, hist_match_2, hist_match_threshold )
    prd_face_bgr = prd_face_bgr.astype(dtype=np.float32)
    return prd_face_bgr

def get_seamlessmatch(out_img, img_face_mask_a, img_bgr_uint8):
    img_face_seamless_mask_a = None
    for i in range(1,10):
        a = img_face_mask_a > i / 10.0
        if len(np.argwhere(a)) == 0:
            continue
        img_face_seamless_mask_a = img_face_mask_a.copy()
        img_face_seamless_mask_a[a] = 1.0
        img_face_seamless_mask_a[img_face_seamless_mask_a <= i / 10.0] = 0.0
        break

    try:
        seam_img = (out_img*255).astype(np.uint8)
        seam_mask = (img_face_seamless_mask_a*255).astype(np.uint8)
        l,t,w,h = cv2.boundingRect( (seam_mask ))
        s_maskcoord = (int(l+w/2), int(t+h/2))
        out_img = cv2.seamlessClone(seam_img, img_bgr_uint8, 
            seam_mask, s_maskcoord , cv2.NORMAL_CLONE )
        out_img = (out_img / 255.0).astype(dtype=np.float32) 
    except Exception as e:
        print ("Seamless fail")
    return out_img

def if_aftereffect(mode, color_transfer_mode, motion_blur_power, 
                    blursharpen_amount, image_denoise_power, 
                    bicubic_degrade_power):
    color_transfer_needed = ('seamless' in mode and color_transfer_mode != 0)
    histmatch_needed = (mode == 'seamless-hist-match')
    motionblur_needed = (motion_blur_power != 0)
    blursharpen_needed = (blursharpen_amount != 0)
    imagedenoise_needed = (image_denoise_power != 0)
    bicubicdegrade_needed = (bicubic_degrade_power != 0)
    return color_transfer_needed or histmatch_needed or \
           motionblur_needed or blursharpen_needed or \
           imagedenoise_needed or bicubicdegrade_needed

def get_motionblur(out_face_bgr, frame_info, motion_blur_power, use_sr=False):
    cfg_mp = motion_blur_power / 100.0
    k_size = int(frame_info.motion_power*cfg_mp)
    if k_size >= 1:
        k_size = np.clip (k_size+1, 2, 50)
        if use_sr:
            k_size *= 2
        out_face_bgr = imagelib.LinearMotionBlur (out_face_bgr, k_size , frame_info.motion_deg)
    return out_face_bgr

def get_imagedenoise(img_bgr, image_denoise_power):
    n = image_denoise_power
    while n > 0:
        img_bgr_denoised = cv2.medianBlur(img_bgr, 5)
        if int(n / 100) != 0:
            img_bgr = img_bgr_denoised
        else:
            pass_power = (n % 100) / 100.0
            img_bgr = img_bgr*(1.0-pass_power)+img_bgr_denoised*pass_power
        n = max(n-10,0)
    return img_bgr

def get_bicubicdegrade(img_bgr, bicubic_degrade_power):
    img_size = (img_bgr.shape[1], img_bgr.shape[0])
    p = 1.0 - bicubic_degrade_power / 101.0
    img_bgr_downscaled = cv2.resize (img_bgr, 
        (int(img_size[0]*p), int(img_size[1]*p )), interpolation=cv2.INTER_CUBIC)
    img_bgr = cv2.resize (img_bgr_downscaled, img_size, interpolation=cv2.INTER_CUBIC)
    return img_bgr