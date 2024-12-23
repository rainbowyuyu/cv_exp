# rainbow_yu cv_exp.cv_unet.eval 🐋✨

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
    y_pred = (y_pred > 0.5).astype(np.uint8)  # Thresholding for binary segmentation

    # 计算混淆矩阵和分类报告
    print(confusion_matrix(y_val.flatten(), y_pred.flatten()))
    print(classification_report(y_val.flatten(), y_pred.flatten()))

if __name__ == "__main__":
    # 替换为DRIVE数据集的路径
    image_dir = 'DRIVE/test/images'
    mask_dir = 'DRIVE/test/1st_manual'
    evaluate_model(image_dir, mask_dir)


