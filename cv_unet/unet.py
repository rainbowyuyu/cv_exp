# rainbow_yu cv_exp.cv_unet.unet 🐋✨

import tensorflow as tf
from tensorflow.keras import layers


class Unet(tf.keras.Model):
    """
    U-Net 模型实现

    参数：
    - data_format: 'channels_first' 或 'channels_last'
    - classes: 输出类别数（通常是1，对应二分类）
    - transpose_conv: 是否使用反卷积进行上采样
    """

    def __init__(self, data_format='channels_last', classes=1, transpose_conv=False, name='unet'):
        super(Unet, self).__init__(name=name)

        self.concat_axis = 3 if data_format == 'channels_last' else 1

        self.e1 = self._encode_block([32, 32], stage=1, data_format=data_format)
        self.e2 = self._encode_block([64, 64], stage=2, data_format=data_format)
        self.e3 = self._encode_block([128, 128], stage=3, data_format=data_format)
        self.e4 = self._encode_block([256, 256], stage=4, data_format=data_format)

        self.d4 = self._decode_block([512, 512, 256], stage=4, data_format=data_format, transpose_conv=transpose_conv)
        self.d3 = self._decode_block([256, 256, 128], stage=3, data_format=data_format, transpose_conv=transpose_conv)
        self.d2 = self._decode_block([128, 128, 64], stage=2, data_format=data_format, transpose_conv=transpose_conv)
        self.d1 = self._decode_block([64, 64, 32], stage=1, data_format=data_format, transpose_conv=transpose_conv)

        self.conv_output = layers.Conv2D(classes, (1, 1), data_format=data_format, name='conv_output')

    def _encode_block(self, filters, stage, data_format):
        filters1, filters2 = filters
        conv_name_base = 'encode' + str(stage) + '_conv'
        bn_name_base = 'encode' + str(stage) + '_bn'

        bn_axis = 3 if data_format == 'channels_last' else 1

        return tf.keras.Sequential([
            layers.Conv2D(filters1, (3, 3), padding='same', data_format=data_format, name=conv_name_base + '2a'),
            layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'),
            layers.ReLU(),
            layers.Conv2D(filters2, (3, 3), padding='same', data_format=data_format, name=conv_name_base + '2b'),
            layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'),
            layers.ReLU(),
            layers.MaxPooling2D(data_format=data_format)
        ])

    def _decode_block(self, filters, stage, data_format, transpose_conv=False):
        filters1, filters2, filters3 = filters
        bn_axis = 3 if data_format == 'channels_last' else 1

        return tf.keras.Sequential([
            layers.Conv2D(filters1, (3, 3), padding='same', data_format=data_format),
            layers.BatchNormalization(axis=bn_axis),
            layers.ReLU(),
            layers.Conv2D(filters2, (3, 3), padding='same', data_format=data_format),
            layers.BatchNormalization(axis=bn_axis),
            layers.ReLU(),
            layers.Conv2DTranspose(filters3, (3, 3), strides=(2, 2), padding='same', data_format=data_format)
        ])

    def call(self, inputs):
        e1x = self.e1(inputs)
        e2x = self.e2(e1x)
        e3x = self.e3(e2x)
        e4x = self.e4(e3x)

        d4x = self.d4(e4x)
        d3x = self.d3(d4x)
        d2x = self.d2(d3x)
        d1x = self.d1(d2x)

        return self.conv_output(d1x)
