import numpy as np
import torch
import linecache
import torch.utils.data as data

class ESImagenet_Dataset(data.Dataset):
    # 初始化数据集对象
    def __init__(self, mode, data_set_path='/data/dvsimagenet/'):
        super().__init__()
        self.mode = mode  # 指定模式（训练或测试）
        self.filenames = []  # 文件名列表
        self.trainpath = data_set_path + 'train'  # 训练数据路径
        self.testpath = data_set_path + 'val'  # 测试数据路径
        self.traininfotxt = data_set_path + 'trainlabel.txt'  # 训练数据标签文件
        self.testinfotxt = data_set_path + 'vallabel.txt'  # 测试数据标签文件
        self.formats = '.npz'  # 文件格式
        if mode == 'train':
            self.path = self.trainpath
            trainfile = open(self.traininfotxt, 'r')  # 打开训练标签文件
            for line in trainfile:
                filename, classnum, a, b = line.split()  # 解析每行数据
                realname, sub = filename.split('.')
                self.filenames.append(realname + self.formats)  # 添加文件名到列表
        else:
            self.path = self.testpath
            testfile = open(self.testinfotxt, 'r')  # 打开测试标签文件
            for line in testfile:
                filename, classnum, a, b = line.split()  # 解析每行数据
                realname, sub = filename.split('.')
                self.filenames.append(realname + self.formats)  # 添加文件名到列表

    # 获取数据集中一个样本
    def __getitem__(self, index):
        if self.mode == 'train':
            info = linecache.getline(self.traininfotxt, index+1)  # 从文件中获取训练数据信息
        else:
            info = linecache.getline(self.testinfotxt, index+1)  # 从文件中获取测试数据信息
        filename, classnum, a, b = info.split()
        realname, sub = filename.split('.')
        filename = realname + self.formats
        filename = self.path + '/' + filename
        classnum = int(classnum)
        a = int(a)
        b = int(b)
        datapos = np.load(filename)['pos'].astype(np.float64)  # 加载正事件数据
        dataneg = np.load(filename)['neg'].astype(np.float64)  # 加载负事件数据

        dy = (254 - b) // 2
        dx = (254 - a) // 2
        input = torch.zeros([2, 8, 256, 256])  # 创建输入张量

        x = datapos[:, 0] + dx
        y = datapos[:, 1] + dy
        t = datapos[:, 2] - 1
        input[0, t, x, y] = 1  # 设置正事件数据点

        x = dataneg[:, 0] + dx
        y = dataneg[:, 1] + dy
        t = dataneg[:, 2] - 1
        input[1, t, x, y] = 1  # 设置负事件数据点

        reshape = input[:, :, 16:240, 16:240]  # 裁剪输入张量以移除边缘
        label = torch.tensor([classnum])  # 创建标签张量
        return reshape, label

    # 返回数据集中样本的总数
    def __len__(self):
        return len(self.filenames)