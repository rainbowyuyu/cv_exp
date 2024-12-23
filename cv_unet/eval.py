# rainbow_yu cv_exp.cv_unet.eval 🐋✨

import tensorflow as tf
from unet import Unet
from process import preprocess_data


def evaluate_model(image_dir, mask_dir, batch_size=32, image_size=(128, 128)):
    # 加载数据集
    val_dataset = preprocess_data(image_dir, mask_dir, image_size=image_size, batch_size=batch_size)

    # 加载训练好的模型，传递 custom_objects 参数
    model = tf.keras.models.load_model('unet_model.h5', custom_objects={'unet': Unet})

    # 评估模型
    loss, accuracy = model.evaluate(val_dataset, verbose=1)

    print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")


if __name__ == '__main__':
    evaluate_model('DRIVE/test/images', 'DRIVE/test/mask')

