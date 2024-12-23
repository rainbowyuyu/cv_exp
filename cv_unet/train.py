# rainbow_yu cv_exp.cv_unet.train 🐋✨

import tensorflow as tf
from process import preprocess_data
from unet import Unet
import os

# 配置路径
image_dir = 'DRIVE/train/images'
mask_dir = 'DRIVE/train/mask'

# 加载数据
X_train, X_test, y_train, y_test, image_datagen, mask_datagen = preprocess_data(image_dir, mask_dir)

# 创建 U-Net 模型
model = Unet(data_format='channels_last', classes=1, transpose_conv=True)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(image_datagen.flow(X_train, y_train), validation_data=(X_test, y_test), epochs=50)

