# -*- coding:utf-8 -*-



import torch as t
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.backends.cudnn as cudnn

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


# 定义网络模型 Alex
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2
            # 经过该层图像大小变为32-7+2*2 / 2 +1，15*15
            # 经3*3最大池化，2步长，图像变为15-3 / 2 + 1， 7*7
            nn.Conv2d(3, 96, 7, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层2，96输入通道，256个卷积核，核大小5*5，步长1，填充2
            # 经过该层图像变为7-5+2*2 / 1 + 1，7*7
            # 经3*3最大池化，2步长，图像变为7-3 / 2 + 1， 3*3
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层3，256输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，256个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            # 256个feature，每个feature 3*3
            nn.Linear(256*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

# 定义网络 VGG
def make_vgg_block(in_channel, out_channel, convs, pool=True):
    net = []

    # 不改变图片尺寸卷积
    net.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
    net.append(nn.BatchNorm2d(out_channel))
    net.append(nn.ReLU(inplace=True))

    for i in range(convs - 1):
        # 不改变图片尺寸卷积
        net.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        net.append(nn.BatchNorm2d(out_channel))
        net.append(nn.ReLU(inplace=True))

    if pool:
        # 2*2最大池化，图片变为w/2 * h/2
        net.append(nn.MaxPool2d(2))

    return nn.Sequential(*net)


# 定义网络模型
class VGG19Net(nn.Module):
    def __init__(self):
        super(VGG19Net, self).__init__()

        net = []

        # 输入32*32，输出16*16
        net.append(make_vgg_block(3, 64, 2))

        # 输出8*8
        net.append(make_vgg_block(64, 128, 2))

        # 输出4*4
        net.append(make_vgg_block(128, 256, 4))

        # 输出2*2
        net.append(make_vgg_block(256, 512, 4))

        # 无池化层，输出保持2*2
        net.append(make_vgg_block(512, 512, 4, False))

        self.cnn = nn.Sequential(*net)

        self.fc = nn.Sequential(
            # 512个feature，每个feature 2*2
            nn.Linear(512*2*2, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


# 定义网络模型
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


'''
定义网络结构
'''
class ResNet34(nn.Module):
    def __init__(self, block):
        super(ResNet34, self).__init__()

        # 初始卷积层核池化层
        self.first = nn.Sequential(
            # 卷基层1：7*7kernel，2stride，3padding，outmap：32-7+2*3 / 2 + 1，16*16
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 最大池化，3*3kernel，1stride（32的原始输入图片较小，不再缩小尺寸），1padding，
            # outmap：16-3+2*1 / 1 + 1，16*16
            nn.MaxPool2d(3, 1, 1)
        )

        # 第一层，通道数不变
        self.layer1 = self.make_layer(block, 64, 64, 3, 1)

        # 第2、3、4层，通道数*2，图片尺寸/2
        self.layer2 = self.make_layer(block, 64, 128, 4, 2)  # 输出8*8
        self.layer3 = self.make_layer(block, 128, 256, 6, 2)  # 输出4*4
        self.layer4 = self.make_layer(block, 256, 512, 3, 2)  # 输出2*2

        self.avg_pool = nn.AvgPool2d(2)  # 输出512*1
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, in_channels, out_channels, block_num, stride):
        layers = []

        # 每一层的第一个block，通道数可能不同
        layers.append(block(in_channels, out_channels, stride))

        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 卷积层
        self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，6个卷积核，核大小5*5
            # 经过该层图像大小变为32-5+1，28*28
            # 经2*2最大池化，图像变为14*14
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 卷积层2，6输入通道，16个卷积核，核大小5*5
            # 经过该层图像变为14-5+1，10*10
            # 经2*2最大池化，图像变为5*5
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            # 16个feature，每个feature 5*5
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

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

    # 样本批次训练的损失函数值的和
    train_loss = 0

    # 识别正确的样本数
    ok = 0

    for i, data in enumerate(train_data_load, 0):
        img, label = data
        # img, label = img.cuda(), label.cuda()

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
        # img, label = img.cuda(), label.cuda()

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
    classes = ('plane', 'car', 'bird', 'cat')

    show = ToPILImage()

    data, label = dataset[index]
    print('img is a ', classes[label])
    show((data + 1) / 2).resize((100, 100)).show()


def main():
    # 训练超参数设置，可通过命令行设置
    parser = argparse.ArgumentParser(description='PyTorch CIFA10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    # train_data = tv.datasets.CIFAR10(root=ROOT, train=True, download=True, transform=transform)
    # test_data = tv.datasets.CIFAR10(root=ROOT, train=False, download=False, transform=transform)

    train_data = tv.datasets.ImageFolder("/home/sun/work/python/blast_furnace/CNN/demo/data_classed",
                                                transform=transform
                                                )
    test_data = tv.datasets.ImageFolder("/home/sun/work/python/blast_furnace/CNN/demo/data_classed",
                                                transform=transform
                                                )
    # print(test_data[0])
    train_load = t.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=WORKERS)
    test_load = t.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=WORKERS)

    net = LeNet()
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