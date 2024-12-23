# rainbow_yu cv_exp.cv_unet.train 🐋✨

import tensorflow as tf
from unet import Unet
from process import preprocess_data


def train_model(image_dir, mask_dir, batch_size=32, epochs=10, image_size=(128, 128)):
    # 加载数据集
    train_dataset = preprocess_data(image_dir, mask_dir, image_size=image_size, batch_size=batch_size)

    # 创建模型
    model = Unet(classes=1)

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_dataset, epochs=epochs, verbose=1)

    # 保存模型
    model.save('unet_model.h5')
    print("Model saved to unet_model.h5")


if __name__ == '__main__':
    train_model('DRIVE/train/images', 'DRIVE/train/mask')
