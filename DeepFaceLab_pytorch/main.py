import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    class fixPathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    videoed_parser = subparsers.add_parser("videoed", help="Video processing.").add_subparsers()

    def process_videoed_extract_video(arguments):
        from mainscripts import videoed
        videoed.extract_video (arguments.input_file, arguments.output_dir, arguments.output_ext, arguments.fps)
    p = videoed_parser.add_parser( "extract-video", help="Extract images from video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory, where the extracted images will be stored.")
    p.add_argument('--output-ext', dest="output_ext", default="jpg", help="Image format (extension) of output files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="How many frames of every second of the video will be extracted. Default - full fps.")
    p.set_defaults(func=process_videoed_extract_video)

    def process_extract(arguments):
        from mainscripts import extract
        extract.extract(
            input_dir    = arguments.input_dir,
            output_dir   = arguments.output_dir,
            input_ext    = arguments.input_ext,
            output_debug = arguments.output_debug,
            face_type    = arguments.face_type,
            image_size   = arguments.image_size,
            conf         = arguments.confidence_level,
            jpeg_quality = arguments.jpeg_quality,
            detector     = arguments.detector,
            gpu          = arguments.device,
        )

    p = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory to be processed.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory, where the extracted images will be stored.")
    p.add_argument('--input-ext', dest="input_ext", default="jpg", help="Image format (extension) of input files.")
    p.add_argument('--output-debug', action="store_true", dest="output_debug", default=False, help="Writes debug images to <output-dir>_debug\ directory.")
    p.add_argument('--face-type', dest="face_type", default="whole_face", choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'])
    p.add_argument('--image-size', type=int, dest="image_size", default=512, help="Output image size.")
    p.add_argument('--confidence-level', type=float, dest="confidence_level", default=0.8, help="Confidence level req. for face bbox")
    p.add_argument('--jpeg-quality', type=int, dest="jpeg_quality", default=90, help="Jpeg quality.")    
    p.add_argument('--detector', dest="detector", default="fa-2d", help="Type of landmark extractor")
    p.add_argument('--device', dest="device", default="cuda:0", help="Type of device used for extraction.")
    p.set_defaults (func=process_extract)

    def process_weightconv(arguments):
        from mainscripts import convert_to_torch
        convert_to_torch.converter(
            origin_path = arguments.origin_dir, 
            torch_path  = arguments.torch_dir, 
            config_name = arguments.config_name, 
            model_name  = arguments.model_name
        )

    p = subparsers.add_parser( "converter", help="Convert DFL weights to PyTorch versions")
    p.add_argument('--origin-dir', required=True, action=fixPathAction, dest="origin_dir", help="Original directory of the DFL trained weights")
    p.add_argument('--torch-dir',  required=True, action=fixPathAction, dest="torch_dir",  help="New directory where the torch files will be stored")
    p.add_argument('--config-name', required=True, type=str, default="SAEHD_default_options.dat", dest="config_name", help="Config filename which should be inside of original directory")
    p.add_argument('--name',       dest="model_name", default="new", help="Type of device used for extraction.")
    p.set_defaults (func=process_weightconv)
    

    def process_train(arguments):
        from mainscripts import train
        train.train(
            src_path = arguments.src_dir, 
            dst_path = arguments.dst_dir, 
            model_path = arguments.model_dir, 
            config_path = arguments.config_dir, 
            gpu_idxs = [ int(x) for x in arguments.device.split(',') ] if arguments.device is not None else None, 
            savedmodel_path = arguments.savedmodel_dir
        )

    p = subparsers.add_parser( "train", help="Trainer")
    p.add_argument('--src-dir',    required=True, action=fixPathAction, dest="src_dir",    help="Source directory where source images are stored")
    p.add_argument('--dst-dir',    required=True, action=fixPathAction, dest="dst_dir",   help="Destination directory, where destination images are stored.")
    p.add_argument('--model-dir',  required=True, action=fixPathAction, dest="model_dir",    help="Image format (extension) of input files.")
    p.add_argument('--config-dir', required=True, action=fixPathAction, dest="config_dir", help="Writes debug images to <output-dir>_debug\ directory.")
    p.add_argument('--device', dest="device", default="0", help="Type of device used for extraction.")
    p.add_argument('--savedmodel-dir', default=None, action=fixPathAction, dest="savedmodel_dir")
    p.set_defaults (func=process_train)


    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    arguments = parser.parse_args()
    arguments.func(arguments)