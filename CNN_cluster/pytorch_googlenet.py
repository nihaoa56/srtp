# -*- coding:utf-8 -*-

import torch as t
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.backends.cudnn as cudnn

import numpy as np
import scipy.misc
from PIL import Image

import cv2

import datetime
import argparse


# 样本读取线程数
WORKERS = 4

# 网络参赛保存文件名
PARAS_FN = 'cifar_lenet_params.pkl'

# minist数据存放位置
ROOT = 'data'

# 目标函数
loss_func = nn.CrossEntropyLoss()

# 最优结果
best_acc = 0



#定义一个卷积加一个relu激活函数和一个batchnorm作为一个基本的层结构
def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
     layer = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
          nn.BatchNorm2d(out_channels, eps=1e-3),
          nn.ReLU(True)
     )
     return layer
 
class inception(nn.Module):
     def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
          super(inception, self).__init__()
          #第一条线路
          self.branch1x1 = conv_relu(in_channel, out1_1, 1)
          
          #第二条线路
          self.branch3x3 = nn.Sequential(
               conv_relu(in_channel, out2_1, 1),
               conv_relu(out2_1, out2_3, 3, padding=1)
          )
          
          #第三条线路
          self.branch5x5 = nn.Sequential(
               conv_relu(in_channel, out3_1, 1),
               conv_relu(out3_1, out3_5, 5, padding=2)
          )
          
          #第四条线路
          self.branch_pool = nn.Sequential(
               nn.MaxPool2d(3, stride=1, padding=1),
               conv_relu(in_channel, out4_1, 1)
          )
     def forward(self, x):
          f1 = self.branch1x1(x)
          f2 = self.branch3x3(x)
          f3 = self.branch5x5(x)
          f4 = self.branch_pool(x)
          output = torch.cat((f1, f2, f3, f4), dim=1)
          return output
        
class googlenet(nn.Module):
     def __init__(self, in_channel, num_classes, verbose=False):
          super(googlenet, self).__init__()
          self.verbose = verbose
          
          self.block1 = nn.Sequential(
               conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
               nn.MaxPool2d(3, 2)
          )
          self.block2 = nn.Sequential(
               conv_relu(64, 64, kernel=1),
               conv_relu(64, 192, kernel=3, padding=1),
               nn.MaxPool2d(3, 2)
          )
          self.block3 = nn.Sequential(
               inception(192, 64, 96, 128, 16, 32, 32),
               inception(256, 128, 128, 192, 32, 96, 64),
               nn.MaxPool2d(3, 2)
          )
          self.block4 = nn.Sequential(
               inception(480, 192, 96, 208, 16, 48, 64),
               inception(512, 160, 112, 224, 24, 64, 64),
               inception(512, 128, 128, 256, 24, 64, 64),
               inception(512, 112, 144, 288, 32, 64, 64),
               inception(528, 256, 160, 320, 32, 128, 128),
               nn.MaxPool2d(3, 2)
          )
          self.block5 = nn.Sequential(
               inception(832, 256, 160, 320, 32, 128, 128),
               inception(832, 384, 182, 384, 48, 128, 128),
               nn.AvgPool2d(2)
          )
          
          self.classifier = nn.Linear(1024, num_classes)
          
     def forward(self, x):
          x = self.block1(x)
          if self.verbose:
               print('block 1 output: {}'.format(x.shape))
          x = self.block2(x)
          if self.verbose:
               print('block 2 output: {}'.format(x.shape))
          x = self.block3(x)
          if self.verbose:
               print('block 3 output: {}'.format(x.shape))
          x = self.block4(x)
          if self.verbose:
               print('block 4 output: {}'.format(x.shape))
          x = self.block5(x)
          if self.verbose:
               print('block 5 output: {}'.format(x.shape))
          
          x = x.view(x.shape[0], -1)
          x = self.classifier(x)
          return x


'''
训练并测试网络
net：网络模型
train_data_load：训练数据集
optimizer：优化器
epoch：第几次训练迭代
log_interval：训练过程中损失函数值和准确率的打印频率
'''
def net_train(net, train_data_load, optimizer, epoch, log_interval):
    net.train()

    begin = datetime.datetime.now()

    # 样本总数
    total = len(train_data_load.dataset)
    print(train_data_load.dataset)

    # 样本批次训练的损失函数值的和
    train_loss = 0

    # 识别正确的样本数
    ok = 0

    for i, data in enumerate(train_data_load, 0):
        img, label = data
        img, label = img.cuda(), label.cuda()

        # print(img)


        optimizer.zero_grad()

        outs = net(img)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()

        # 累加损失值和训练样本数
        train_loss += loss.item()
        # total += label.size(0)

        _, predicted = t.max(outs.data, 1)
        # 累加识别正确的样本数
        ok += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            # 训练结果输出

            # 损失函数均值
            loss_mean = train_loss / (i + 1)

            # 已训练的样本数
            traind_total = (i + 1) * len(label)

            # 准确度
            acc = 100. * ok / traind_total

            # 一个迭代的进度百分比
            progress = 100. * traind_total / total

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, traind_total, total, progress, loss_mean, acc))

    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)


'''
用测试集检查准确率
'''
def net_test(net, test_data_load, epoch):
    net.eval()

    ok = 0

    for i, data in enumerate(test_data_load):
        img, label = data
        img, label = img.cuda(), label.cuda()

        outs = net(img)
        _, pre = t.max(outs.data, 1)
        ok += (pre == label).sum()

    acc = ok.item() * 100. / (len(test_data_load.dataset))
    print('EPOCH:{}, ACC:{}\n'.format(epoch, acc))

    global best_acc
    if acc > best_acc:
        best_acc = acc


'''
显示数据集中一个图片
'''
def img_show(dataset, index):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    show = ToPILImage()

    data, label = dataset[index]
    print('img is a ', classes[label])
    show((data + 1) / 2).resize((100, 100)).show()


def main():
    # 训练超参数设置，可通过命令行设置
    parser = argparse.ArgumentParser(description='PyTorch CIFA10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='If train the Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # 图像数值转换，ToTensor源码注释
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
        Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        """
    # 归一化把[0.0, 1.0]变换为[-1,1], ([0, 1] - 0.5) / 0.5 = [-1, 1]
    transform = tv.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 定义数据集
    train_data = tv.datasets.CIFAR10(root=ROOT, train=True, download=True, transform=transform)
    test_data = tv.datasets.CIFAR10(root=ROOT, train=False, download=False, transform=transform)

    train_load = t.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=WORKERS)
    test_load = t.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=WORKERS)

    # img_show(train_data,1)
    data, label = train_data[1]
    data = data.numpy()
    np.savetxt("new.csv", data, delimiter=',')


    net = googlenet(3,)
    print(net)


    # 如果不训练，直接加载保存的网络参数进行测试集验证
    if args.no_train:
        net.load_state_dict(t.load(PARAS_FN))
        net_test(net, test_load, 0)
        return

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        net_train(net, train_load, optimizer, epoch, args.log_interval)

        # 每个epoch结束后用测试集检查识别准确度
        net_test(net, test_load, epoch)

    end_time = datetime.datetime.now()

    global best_acc
    print('CIFAR10 pytorch LeNet Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size, args.lr, best_acc))
    print('train spend time: ', end_time - start_time)

    if args.save_model:
        t.save(net.state_dict(), PARAS_FN)


if __name__ == '__main__':
    main()

 
          
