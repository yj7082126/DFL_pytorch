import numpy as np
import cv2

def save_image(tensors, img_path):
    n, c, h, w  = tensors["Target SRC"].shape

    tmp_list = []
    for _, tensor in tensors.items():
        tmp = tensor.permute(0,2,3,1).reshape(-1, w, c)
        tmp = tmp.detach().cpu().numpy()
        tmp = (tmp * 255).astype(np.uint8)
        tmp_list.append(tmp)

    tmp_array = np.concatenate(tmp_list, axis=1)
    cv2.imwrite(str(img_path), tmp_array)

def write_losses(writer, losses, curr_iter):
    for name, loss in losses.items():
        writer.add_scalar(name, loss, curr_iter)


def write_images(writer, images, debug_path, curr_iter):
    for name, image in images.items():
        writer.add_images(name,  image.flip([1]), global_step=curr_iter)

    img_path = debug_path.joinpath(f"{curr_iter:06d}.jpg")
    save_image(images, img_path)