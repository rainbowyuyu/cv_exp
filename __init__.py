# rainbow_yu cv_exp 🐋✨
# 计算机视觉实验难点汇总

__version__ = '0.5.0'

__all__ = (
    "cv_mnist",  # cv_exp4
    "cv_cnn",  # cv_exp5
    "cv_rcnn",  # cv_exp6
    "cv_unet_tf",  # cv_exp7 tensorflow版本
    "cv_unet_pt"  # cv_exp7 pytorch版本
)

from cv_mnist import *
from cv_cnn import *
from cv_rcnn import *
from cv_unet_tf import *
from cv_unet_pt import *
