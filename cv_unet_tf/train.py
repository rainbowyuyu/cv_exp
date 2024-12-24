# rainbow_yu cv_exp.cv_unet_tf.train 🐋✨

import tensorflow as tf
from unet import unet_model
from process import preprocess_data


def train_model(image_dir, mask_dir, epochs=10, batch_size=16):
    # 加载和预处理数据
    X_train, X_val, y_train, y_val = preprocess_data(image_dir, mask_dir)

    # 构建模型
    model = unet_model()

    # 训练模型
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size)

    # 保存模型
    model.save('unet_model.h5')

    return model, history


if __name__ == "__main__":
    # 替换为DRIVE数据集的路径
    image_dir = 'DRIVE/training/images'
    mask_dir = 'DRIVE/training/1st_manual'
    model, history = train_model(image_dir, mask_dir)
