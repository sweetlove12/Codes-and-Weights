import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义加载 CIFAR-10 数据集并进行预处理的函数
def build_dataset(is_train, args):
    # 根据训练状态选择适当的图像变换
    transform = build_transform(is_train, args)

    # 构建 CIFAR-10 数据集的路径
    root = os.path.join(args.data_path, "CIFAR10")
    # 创建 CIFAR-10 数据集对象，自动下载数据集（如果本地不存在）
    dataset = datasets.CIFAR10(root, train=is_train, transform=transform, download=True)

    print(dataset)  # 打印数据集信息，便于调试
    return dataset


# 定义图像预处理变换的函数
def build_transform(is_train, args):
    # 设置 CIFAR-10 数据集的均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # 如果是训练数据，添加数据增强步骤
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪，边缘填充 4 像素
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),  # 将图像转换为 Tensor
            transforms.Normalize(mean, std)  # 归一化处理
        ])
    else:
        # 验证数据不进行数据增强，只进行标准化处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


# 测试函数
if __name__ == "__main__":
    # 假设 args 包含必要的配置项
    class Args:
        data_path = './data'  # 设置数据存放路径
        input_size = 32  # CIFAR-10 图像的输入尺寸为 32x32


    args = Args()

    # 构建训练和验证数据集
    train_dataset = build_dataset(True, args)
    val_dataset = build_dataset(False, args)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 迭代一次训练数据加载器，打印一些图像的形状和标签，验证数据加载是否正确
    for images, labels in train_loader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        break  # 只展示第一批数据

