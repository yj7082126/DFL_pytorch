from pathlib import Path
from tqdm import tqdm
from operator import itemgetter

import numpy as np
import cv2
from face_alignment import FaceAlignment, LandmarksType

from facelib import FaceType, LandmarksProcessor
from DFLIMG import *

def extract(input_dir, output_dir, input_ext="jpg", output_debug=False, 
            face_type="whole_face", image_size=512, conf=0.8,
            jpeg_quality=90,  detector="fa-2d", gpu="cuda:0"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_debug:
        debug_dir = output_dir.parent.joinpath(output_dir.name + "_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

    input_image_paths = sorted(list(input_dir.glob(f"*.{input_ext}")))

    face_type    = FaceType.fromString(face_type)
    if detector == "fa-2d":
        fa = FaceAlignment(LandmarksType._2D, device=gpu)
    else:
        fa = FaceAlignment(LandmarksType._3D, device=gpu)

    faces_found = 0
    for filepath in tqdm(input_image_paths):
        image = cv2.imread(str(filepath))
        image_debug = image.copy()

        result = fa.face_detector.detect_from_image(image)
        result = [x for x in result if x[-1] > conf]
        result = sorted(result, key=itemgetter(0))

        rects  = [list(x.astype(np.int32)[:4]) for x in result]
        landmarks = fa.get_landmarks_from_image(image, rects)

        if landmarks:
            for i, (rect, image_landmarks) in enumerate(zip( rects, landmarks )):
                faces_found += 1
                image_to_face_mat = LandmarksProcessor.get_transform_mat (image_landmarks, image_size, face_type)

                face_image = cv2.warpAffine(image, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)
                face_image_landmarks = LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat)

                landmarks_bbox = LandmarksProcessor.transform_points ( [ 
                    (0,0), (0,image_size-1), (image_size-1, image_size-1), (image_size-1,0) 
                ], image_to_face_mat, True)

                output_filepath = output_dir.joinpath(f"{filepath.stem}_{i}.jpg")
                cv2.imwrite(str(output_filepath), face_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]) 

                dflimg = DFLJPG.load(output_filepath)
                dflimg.set_face_type(FaceType.toString(face_type))
                dflimg.set_landmarks(face_image_landmarks.tolist())
                dflimg.set_source_filename(filepath.name)
                dflimg.set_source_rect(rect)
                dflimg.set_source_landmarks(image_landmarks.tolist())
                dflimg.set_image_to_face_mat(image_to_face_mat)
                dflimg.save()

                LandmarksProcessor.draw_rect_landmarks(image_debug, 
                    rects[i], landmarks[i], face_type, 
                    face_size=image_size, transparent_mask=True)
            
        if output_debug:
            debug_filepath = debug_dir.joinpath(f"{filepath.stem}.jpg")
            cv2.imwrite(str(debug_filepath), image_debug, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]) 

    print(f"Total Faces : {faces_found}")