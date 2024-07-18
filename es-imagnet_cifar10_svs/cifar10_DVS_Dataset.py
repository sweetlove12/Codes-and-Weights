import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import os


class cifar10_DVS(data.Dataset):
    # 初始化数据集类
    def __init__(self, path='/data/CIFAR10-MAT', mode='train', step=2):
        self.mode = mode  # 模式：训练或测试
        self.filenames = []  # 存储文件名的列表
        self.trainpath = path + '/train/'  # 训练数据的路径
        self.testpath = path + '/test/'  # 测试数据的路径
        self.formats = '.mat'  # 文件格式为.mat
        self.step = step  # 在加载时对时间序列数据进行下采样的步长

        # 根据模式确定路径，并遍历该路径下的所有文件，将文件路径存储到 filenames 列表中
        if mode == 'train':
            self.path = self.trainpath
            for file in os.listdir(self.trainpath):
                self.filenames.append(self.trainpath + file)
        else:
            self.path = self.testpath
            for file in os.listdir(self.testpath):
                self.filenames.append(self.testpath + file)

        self.num_sample = int(len(self.filenames))  # 数据集样本总数

    # 获取索引对应的数据项
    def __getitem__(self, index):
        image = 0
        label = 0
        try:
            # 从.mat 文件中加载数据
            data = sio.loadmat(self.filenames[index])
            image, label, name = data['frame_Data'], data['label'], data['name']
            image = image.astype(np.float32)  # 将图像数据转换为 float32 类型
            label = label.astype(np.float32)  # 将标签数据转换为 float32 类型
            # 按照设定的步长对时间序列数据进行下采样
            image = torch.from_numpy(image[:, 0:20:self.step, :, :]).float()
            label = torch.from_numpy(label[0, :]).long()  # 将标签转换为长整型
        except Exception as e:
            print(e)  # 打印异常信息
            # 如果加载失败，返回预定义的空图像和标签
            image = torch.zeros([2, 20 // self.step, 128, 128]).float()
            label = torch.zeros([1]).long()
        return image, label

    # 返回数据集中的样本总数
    def __len__(self):
        return self.num_sample

