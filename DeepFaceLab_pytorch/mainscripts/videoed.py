from pathlib import Path

import numpy as np
import cv2
import ffmpeg

def extract_video(input_file, output_dir, ext="jpg", fps=None):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fps is None:
        fps = cv2.VideoCapture(str(input_file)).get(cv2.CAP_PROP_FPS)

    output_args = [
        ffmpeg.input(str(input_file)),
        str(output_dir.joinpath(f"%05d.{ext}"))
    ]

    output_kwargs = {
        "pix_fmt" : "rgb24", 
        "r" : str(fps),
        "q:v" : "2"
    }

    job = (ffmpeg.output(*output_args, **output_kwargs).overwrite_output())

    try:
        job = job.run()
    except:
        print(f"FFMPEG FAIL : {str(job.compile())}")
    
    frames_found = len(list(output_dir.glob(f"*.{ext}")))
    print(f"Frames : {frames_found}")

def video_from_sequence(input_dir, output_file, ext="jpg", fps=None):
    input_path = Path(input_dir)
    output_file_path = Path(output_file)

    if not input_path.exists():
        print("input_dir not found.")
        return

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        return

    if fps is None:
        fps = 30

    input_image_paths = sorted(list(output_path.glob(f"*.{ext}")))

    output_args = [
        ffmpeg.input('pipe:', format='image2pipe', r=fps), 
        str (output_file_path)
    ]

    output_kwargs = {
        "c:v": "libx264",
        "b:v": "16M",
        "pix_fmt": "yuv420p"
    }

    job = ( ffmpeg.output(*output_args, **output_kwargs).overwrite_output() )

    try:
        job_run = job.run_async(pipe_stdin=True)

        for image_path in input_image_paths:
            with open (image_path, "rb") as f:
                image_bytes = f.read()
                job_run.stdin.write (image_bytes)

        job_run.stdin.close()
        job_run.wait()
    except:
        print("ffmpeg fail, job commandline:" + str(job.compile()) )    