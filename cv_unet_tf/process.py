# rainbow_yu cv_exp.cv_unet_tf.process 🐋✨

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image


def load_images(image_dir, mask_dir):
    images = []
    masks = []

    for file_name in os.listdir(image_dir):
        if file_name.endswith(".tif"):
            img_path = os.path.join(image_dir, file_name)
            mask_path = os.path.join(mask_dir, file_name.replace('training', 'manual1').replace('.tif', '.gif'))

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"Mask not found: {mask_path}")
                continue

            # Load image using cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img = cv2.resize(img, (256, 256))  # Resize image to consistent size
            img = np.expand_dims(img, axis=-1) / 255.0  # Normalize and add channel dimension

            # Load mask using PIL to handle GIF format
            mask = Image.open(mask_path).convert('L')  # Convert to grayscale ('L')
            mask = np.array(mask)
            if mask is None:
                print(f"Failed to read mask: {mask_path}")
                continue
            mask = cv2.resize(mask, (256, 256))  # Resize mask to consistent size
            mask = np.expand_dims(mask, axis=-1) / 255.0  # Normalize

            images.append(img)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks


def preprocess_data(image_dir, mask_dir, test_size=0.2):
    """
    Split data into train and validation sets.
    """
    images, masks = load_images(image_dir, mask_dir)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, train_size=1-test_size,test_size=test_size, random_state=42)

    return X_train, X_val, y_train, y_val


