# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:05:58 2019

@author: Administrator
"""
# http://blog.itpub.net/31562039/viewspace-2565264/
'''
用例2：物体图像分类

在这个用例中，我们将在PyTorch中创建卷积神经网络(CNN)架构，利用流行的CIFAR-10数据集进行
物体图像分类，此数据集也包含在torchvision包中。定义和训练模型的整个过程将与以前的用例
相同，唯一的区别只是在网络中引入了额外的层。
'''
## load the dataset
import torch
print(torch.cuda.is_available())
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch import nn
import numpy as np
import time

start = time.clock()

_tasks = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

cifar = CIFAR10('data', train=True, download=True, transform=_tasks) # 加载并转换数据集

## create training and validation split
split = int(0.8 * len(cifar))
index_list = list(range(len(cifar)))
train_idx, valid_idx = index_list[:split], index_list[split:]
## create training and validation sampler objects
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)
## create iterator objects for train and valid datasets
trainloader = DataLoader(cifar, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(cifar, batch_size=256, sampler=val_sampler)


'''
我们将创建三个用于低层特征提取的卷积层、三个用于最大信息量提取的池化层和两个用于线性分类
的线性层。
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        ## define the layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 1024) ## reshaping
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
model = Model()


#定义损失函数和优化器：
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, 
                      momentum = 0.9, nesterov = True)

## run for 30 Epochs
for epoch in range(1, 31):
    train_loss, valid_loss = [], []
    ## training part
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    ## evaluation part
    model.eval()
    for data, target in validloader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())


#完成了模型的训练之后，即可在验证数据基础上进行预测。
## dataloader for validation dataset
dataiter = iter(validloader)
data, labels = dataiter.next()
output = model(data)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
print ("Actual:", labels[:10])
print ("Predicted:", preds[:10])
#Actual: ['truck', 'truck', 'truck', 'horse', 'bird', 'truck', 'ship', 'bird', 'deer', 'bird']
#Pred:   ['truck', 'automobile', 'automobile', 'horse', 'bird', 'airplane', 'ship', 'bird', 'deer', 'bird']

end = time.clock()
time = end - start
print("耗时：{}s".format(time))