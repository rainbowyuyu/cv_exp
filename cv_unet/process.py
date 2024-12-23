# rainbow_yu cv_exp.cv_unet.process 🐋✨

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images(image_dir, mask_dir, image_size=(256, 256)):
    """加载图像和掩码"""
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]

    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img = image.load_img(img_path, target_size=image_size)
        img = image.img_to_array(img) / 255.0  # 归一化

        mask = image.load_img(mask_path, target_size=image_size, color_mode='grayscale')
        mask = image.img_to_array(mask) / 255.0  # 归一化

        images.append(img)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks


def preprocess_data(image_dir, mask_dir, image_size=(256, 256), test_size=0.2):
    """预处理数据并划分训练集和测试集"""
    images, masks = load_images(image_dir, mask_dir, image_size)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=42)

    # 数据增强
    data_gen_args = dict(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(X_train, augment=True, seed=1)
    mask_datagen.fit(y_train, augment=True, seed=1)

    return X_train, X_test, y_train, y_test, image_datagen, mask_datagen


