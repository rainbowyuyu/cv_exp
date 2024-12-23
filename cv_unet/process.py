# rainbow_yu cv_exp.cv_unet.process 🐋✨

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_data(image_dir, mask_dir, image_size=(128, 128)):
    """
    加载图像和掩码数据，进行尺寸调整并转换为 NumPy 数组。

    参数：
    - image_dir: 图像文件夹路径
    - mask_dir: 掩码文件夹路径
    - image_size: 目标图像尺寸

    返回：
    - images: 处理后的图像数据
    - masks: 处理后的掩码数据
    """
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    images = []
    masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        # 加载图像并调整尺寸
        img = load_img(img_path, target_size=image_size)
        img = img_to_array(img) / 255.0  # 归一化至[0, 1]

        # 加载掩码并调整尺寸
        mask = load_img(mask_path, target_size=image_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0  # 归一化至[0, 1]

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


def preprocess_data(image_dir, mask_dir, image_size=(128, 128), batch_size=32):
    """
    预处理数据并生成批量数据。

    参数：
    - image_dir: 图像文件夹路径
    - mask_dir: 掩码文件夹路径
    - image_size: 图像的目标尺寸
    - batch_size: 批次大小

    返回：
    - tf.data.Dataset: 处理后的数据集
    """
    images, masks = load_data(image_dir, mask_dir, image_size)

    # 将数据转换为 TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
