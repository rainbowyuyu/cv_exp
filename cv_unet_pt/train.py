# rainbow_yu cv_exp.cv_unet_pt.train 🐋✨

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from unet import UNet
import copy

train_data = np.load("data_done/training_dataset.npy", allow_pickle=True)
test_data = np.load("data_done/test_dataset.npy", allow_pickle=True)


# 数据库加载
class DatasetNew(Dataset):
    def __init__(self, data):
        self.len = len(data)
        self.x_data = torch.from_numpy(np.array(list(map(lambda x: x[0], data)), dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(list(map(lambda x: x[1], data)))).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 数据库dataloader
Train_dataset = DatasetNew(train_data)
Test_dataset = DatasetNew(test_data)
dataloader = DataLoader(Train_dataset, shuffle=True)
testloader = DataLoader(Test_dataset, shuffle=True)
# 训练设备选择GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型初始化
model = UNet(3, 1)

model.to(device)

# 损失函数选择
# criterion = torch.nn.BCELoss()

criterion = torch.nn.MSELoss()
# criterion = torch.nn.CrossEntropyLoss()


criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loss = []
test_loss = []


# 训练函数
def train():
    mloss = []
    for data in dataloader:
        datavalue, datalabel = data
        datavalue, datalabel = datavalue.to(device), datalabel.to(device)
        datalabel_pred = model(datavalue)
        loss = criterion(datalabel_pred, datalabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mloss.append(loss.item())

    epoch_train_loss = torch.mean(torch.Tensor(mloss)).item()
    train_loss.append(epoch_train_loss)
    print("*" * 10, epoch, "*" * 10)
    print('训练集损失:', epoch_train_loss)
    test()


# 测试函数
def test():
    mloss = []
    with torch.no_grad():
        for testdata in testloader:
            testdatavalue, testdatalabel = testdata
            testdatavalue, testdatalabel = testdatavalue.to(device), testdatalabel.to(device)
            testdatalabel_pred = model(testdatavalue)
            loss = criterion(testdatalabel_pred, testdatalabel)
            mloss.append(loss.item())
        epoch_test_loss = torch.mean(torch.Tensor(mloss)).item()
        test_loss.append(epoch_test_loss)
        print('测试集损失', epoch_test_loss)


if __name__ == '__main__':
    bestmodel = None
    bestepoch = None
    bestloss = np.inf

    for epoch in range(1, 11):
        train()
        if test_loss[epoch - 1] < bestloss:
            bestloss = test_loss[epoch - 1]
            bestepoch = epoch
            bestmodel = copy.deepcopy(model)

    print("最佳轮次为:{},最佳损失为:{}".format(bestepoch, bestloss))

    torch.save(model, "../../DRIVE_datasets/model/last.pt")
    torch.save(bestmodel, "../../DRIVE_datasets/model/best.pt")

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
