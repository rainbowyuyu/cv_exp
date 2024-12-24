# rainbow_yu cv_exp.cv_unet_tf.eval 🐋✨

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from unet import unet_model
from process import preprocess_data
import numpy as np


def evaluate_model(image_dir, mask_dir):
    # 加载和预处理数据
    _, X_val, _, y_val = preprocess_data(image_dir, mask_dir)

    # 加载训练好的模型
    model = tf.keras.models.load_model('unet_model.h5')

    # 预测结果
    y_pred = model.predict(X_val)
    y_pred = (y_pred > 0.5).astype(int)  # Thresholding for binary segmentation


if __name__ == "__main__":
    # 替换为DRIVE数据集的路径
    image_dir = 'DRIVE/training/images'
    mask_dir = 'DRIVE/training/1st_manual'
    evaluate_model(image_dir, mask_dir)


