# rainbow_yu cv_exp.cv_unet_pt.read_picture 🐋✨

from PIL import Image
import numpy as np


def read_picture(file_path):
    with Image.open(file_path) as img:
        return np.array(img)


if __name__ == "__main__":
    gif_path = "../../DRIVE_datasets/DRIVE/training/images/21_training.tif"
    img = read_picture(gif_path)
    print(img.shape)
