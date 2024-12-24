# rainbow_yu cv_exp.cv_unet_pt.dataset_operation 🐋✨

import numpy as np
import os
from read_picture import read_picture
import cv2

for name in ['training', 'test']:
    picture_path = rf"../../DRIVE_datasets/DRIVE\{name}\images"
    label_path = rf"../../DRIVE_datasets/DRIVE\{name}\1st_manual"

    picture_names = os.listdir(picture_path)
    label_names = os.listdir(label_path)

    data = []
    label = []

    for d, l in zip(picture_names, label_names):
        dp = os.path.join(picture_path, d)
        lp = os.path.join(label_path, l)

        # 读取图片并调整大小为 512x512
        p = cv2.resize(read_picture(dp), (512, 512)).transpose(2, 0, 1)  # 维度调整为 (C, H, W)
        l = cv2.resize(read_picture(lp), (512, 512)).reshape(1, 512, 512)  # 标签维度调整为 (1, H, W)

        # 归一化处理
        p = (p - np.min(p)) / (np.max(p) - np.min(p))
        l = (l - np.min(l)) / (np.max(l) - np.min(l))

        # 将数据和标签分别添加到列表
        data.append(p)
        label.append(l)

    # 将 data 和 label 转换为 numpy 数组
    data = np.array(data)
    label = np.array(label)

    # 如果需要将 data 和 label 配对，使用 dtype=object 来处理不同形状的数组
    dataset = np.array(list(zip(data, label)), dtype=object)

    np.save(f"../../DRIVE_datasets/data_done/{name}_dataset", dataset)
