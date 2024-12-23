# rainbow_yu cv_exp.cv_unet.eval 🐋✨

import tensorflow as tf
from process import preprocess_data
from unet import Unet
from sklearn.metrics import mean_absolute_error

# 配置路径
image_dir = 'path/to/images'
mask_dir = 'path/to/masks'

# 加载数据
X_train, X_test, y_train, y_test, _, _ = preprocess_data(image_dir, mask_dir)

# 加载训练好的模型
model = tf.keras.models.load_model('unet_model.h5')

# 预测
y_pred = model.predict(X_test)

# 计算评估指标
mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
print("Mean Absolute Error:", mae)

