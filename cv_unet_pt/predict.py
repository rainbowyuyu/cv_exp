# rainbow_yu cv_exp.cv_unet_pt.predict 🐋✨

import torch
import numpy as np
import cv2

dataset = np.load('../../DRIVE_datasets/data_done/test_dataset.npy', allow_pickle=True)

data = dataset[0][0]
label = dataset[0][1]

model = torch.load('../../DRIVE_datasets/model/best.pt').cpu()
# model = torch.load('model/last.pt').cpu()

output = model(torch.Tensor(data.reshape(1, 3, data.shape[-2], data.shape[-1]))).detach().numpy()

cv2.imshow('label', label[0])
cv2.imshow('output', output[0][0])
cv2.waitKey()
