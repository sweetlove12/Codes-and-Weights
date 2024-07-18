import os
import PIL
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# 构建数据集函数，根据训练或验证阶段应用不同的变换
def build_dataset(is_train, args):
    # 调用构建变换的函数获取相应的图像预处理方式
    transform = build_transform(is_train, args)

    # 根据是否是训练阶段来确定数据的目录
    root = os.path.join(args.data_path, "train" if is_train else "val")

    # 使用 ImageFolder 加载数据集，ImageFolder 假设所有的数据按文件夹分类，每个类一个文件夹
    dataset = datasets.ImageFolder(root, transform=transform)

    # 打印数据集信息，便于调试
    print(dataset)

    # 返回构建的数据集
    return dataset


# 构建图像预处理变换的函数
def build_transform(is_train, args):
    # 设定使用的图像均值和标准差，这里使用 ImageNet 的默认值
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # 如果是训练数据，应用一系列的图像增强处理
    if is_train:
        # 使用 TIMM 库的 create_transform 创建训练用的图像变换
        # 包括尺寸调整、颜色抖动、自动增强和随机擦除等
        transform = create_transform(
            input_size=args.input_size,  # 图像输入尺寸
            is_training=True,  # 指定为训练模式
            color_jitter=args.color_jitter,  # 颜色抖动强度
            auto_augment=args.aa,  # 自动增强策略
            interpolation="bicubic",  # 插值方法
            re_prob=args.reprob,  # 随机擦除概率
            re_mode=args.remode,  # 随机擦除模式
            re_count=args.recount,  # 随机擦除次数
            mean=mean,  # 归一化均值
            std=std  # 归一化标准差
        )
        return transform

    # 如果是验证数据，应用标准化和中心裁剪处理
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # 根据设定的尺寸比例调整大小
    )
    t.append(transforms.CenterCrop(args.input_size))  # 中心裁剪到指定尺寸

    t.append(transforms.ToTensor())  # 将图像转换为 Tensor
    t.append(transforms.Normalize(mean, std))  # 应用归一化
    return transforms.Compose(t)  # 将所有变换组合起来

