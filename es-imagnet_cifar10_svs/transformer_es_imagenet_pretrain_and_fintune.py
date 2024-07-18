# 导入所需的库
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_spikformer as lrd
import util.misc as misc
from datasets.es_imagenet_new import ESImagenet_Dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.kd_loss import DistillationLoss

import models
import models_sew

from engine_finetune import train_one_epoch, evaluate


# 用于构建ES-ImageNet数据集的函数
def build_es_imagenet_dataset(is_train, args):
    if is_train:
        dataset = ESImagenet_Dataset(args.data_path, train=True)
    else:
        dataset = ESImagenet_Dataset(args.data_path, train=False)
    return dataset


# 获取命令行参数解析器
def get_args_parser():
    parser = argparse.ArgumentParser("MAE fine-tuning for ES-ImageNet image classification", add_help=False)
    # 定义训练的基本参数
    parser.add_argument("--batch_size", default=64, type=int, help="每个GPU的批处理大小")
    parser.add_argument("--epochs", default=200, type=int, help="训练的轮数")
    parser.add_argument("--accum_iter", default=1, type=int, help="累积梯度迭代次数")
    parser.add_argument("--finetune", default="", help="从检查点微调")
    parser.add_argument("--data_path", default="/path/to/es-imagenet", type=str, help="ES-ImageNet数据集的路径")
    parser.add_argument("--model", default="spikformer_8_384_CAFormer", type=str, help="训练模型的名称")
    parser.add_argument("--model_mode", default="ms", type=str, help="模型模式")
    parser.add_argument("--input_size", default=224, type=int, help="输入图像的尺寸")
    parser.add_argument("--drop_path", type=float, default=0.1, help="drop path概率")
    parser.add_argument("--clip_grad", type=float, default=None, help="梯度裁剪阈值")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--blr", type=float, default=6e-4, help="基础学习率")
    parser.add_argument("--layer_decay", type=float, default=1.0, help="层级学习率衰减")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="循环调度器的最低学习率")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="学习率预热的轮数")
    parser.add_argument("--color_jitter", type=float, default=None, help="颜色抖动")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", help="AutoAugment策略")
    parser.add_argument("--smoothing", type=float, default=0.1, help="标签平滑")
    parser.add_argument("--reprob", type=float, default=0.25, help="随机擦除概率")
    parser.add_argument("--remode", type=str, default="pixel", help="随机擦除模式")
    parser.add_argument("--recount", type=int, default=1, help="随机擦除次数")
    parser.add_argument("--resplit", action="store_true", default=False, help="是否重新分割数据集")
    parser.add_argument("--mixup", type=float, default=0, help="mixup alpha值")
    parser.add_argument("--cutmix", type=float, default=0, help="cutmix alpha值")
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None, help="cutmix的最小和最大比例")
    parser.add_argument("--mixup_prob", type=float, default=1.0, help="执行mixup或cutmix的概率")
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5, help="在启用mixup和cutmix时切换到cutmix的概率")
    parser.add_argument("--mixup_mode", type=str, default="batch", help="应用mixup/cutmix参数的方式")
    parser.add_argument("--global_pool", action="store_true", help="是否使用全局池化")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool", help="使用类token进行分类")
    parser.add_argument("--time_steps", default=1, type=int, help="时间步长")
    parser.add_argument("--nb_classes", default=1000, type=int, help="类别数")
    parser.add_argument("--output_dir", default="/path/to/output_dir", help="保存路径")
    parser.add_argument("--log_dir", default="/path/to/log_dir", help="TensorBoard日志路径")
    parser.add_argument("--device", default="cuda", help="训练/测试使用的设备")
    parser.add_argument("--seed", default=0, type=int, help="随机种子")
    parser.add_argument("--resume", default=None, help="从检查点恢复")
    parser.add_argument("--start_epoch", default=0, type=int, help="起始轮数")
    parser.add_argument("--eval", action="store_true", help="只执行评估")
    parser.add_argument("--dist_eval", action="store_true", default=False, help="启用分布式评估")
    parser.add_argument("--num_workers", default=10, type=int, help="数据加载器的工作线程数")
    parser.add_argument("--pin_mem", action="store_true", help="是否将数据加载器的内存固定")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="不将数据加载器的内存固定")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--kd", action="store_true", default=False, help="启用知识蒸馏")
    parser.add_argument("--teacher_model", default="caformer_b36_in21ft1k", type=str, help="教师模型的名称")
    parser.add_argument("--distillation_type", default="none", choices=["none", "soft", "hard"], type=str, help="蒸馏类型")
    parser.add_argument("--distillation_alpha", default=0.5, type=float, help="蒸馏alpha值")
    parser.add_argument("--distillation_tau", default=1.0, type=float, help="蒸馏温度值")
    parser.add_argument("--world_size", default=1, type=int, help="分布式进程数")
    parser.add_argument("--local-rank", default=-1, type=int, help="本地进程排名")
    parser.add_argument("--dist_on_itp", action="store_true", help="启用ITP分布式")
    parser.add_argument("--dist_url", default="env://", help="设置分布式训练的URL")
    return parser


def main(args):
    # 初始化分布式训练模式
    misc.init_distributed_mode(args)

    # 调试：打印工作目录
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # 设置设备
    device = torch.device(args.device)

    # 设置随机种子以确保结果可重复
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置 CUDA 为 benchmark 模式
    cudnn.benchmark = True

    # 构建训练和验证数据集
    dataset_train = build_es_imagenet_dataset(True, args)
    dataset_val = build_es_imagenet_dataset(False, args)

    # 根据是否启用分布式评估选择适当的采样器
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                            shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. This will slightly alter validation results as extra duplicate entries are added to achieve equal number of samples per-process.")
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank,
                                                              shuffle=True)  # shuffle=True以减少监视偏差
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 如果是主进程且指定了日志目录，则创建日志写入器
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # 创建训练和验证的数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    # 如果启用了 mixup 或 cutmix，配置相应的处理函数
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # 初始化模型，根据模型模式选择不同的模型实现
    if args.model_mode == "ms":
        model = models.__dict__[args.model](kd=args.kd)
    elif args.model_mode == "sew":
        model = models_sew.__dict__[args.model]()
    model.T = args.time_steps

    # 如果指定了微调，则从检查点加载模型
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # 将模型移动到指定的设备
    model.to(device)

    # 获取模型的参数数量
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    # 计算实际的批处理大小
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # 如果没有指定学习率，则根据基础学习率和批处理大小计算
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # 如果是分布式训练，配置分布式数据并行
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # 构建优化器，使用层级学习率衰减
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, layer_decay=args.layer_decay)
    optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=args.lr)
    loss_scaler = NativeScaler()

    # 根据是否使用混合标签，选择合适的损失函数
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # 如果启用了知识蒸馏，配置教师模型和蒸馏损失
    if args.kd:
        teacher_model = None
        if args.distillation_type == "none":
            args.distillation_type = "hard"
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = caformer_b36_in21ft1k(pretrained=True)
        teacher_model.to(device)
        teacher_model.eval()
        criterion = DistillationLoss(criterion, teacher_model, args.distillation_type, args.distillation_alpha,
                                     args.distillation_tau)

    print("criterion = %s" % str(criterion))

    # 加载模型和优化器状态
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # 如果是仅评估模式，执行评估并退出
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    # 打印训练开始信息
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_acc = 0
    best_epoch = 0

    # 执行训练周期
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
                                      args.clip_grad, mixup_fn, log_writer=log_writer, args=args)
        # 定期保存模型
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            print("Saving model at epoch:", epoch)
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch)

        # 执行验证并打印结果
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")
        if args.output_dir and test_stats["acc1"] > best_acc:
            print("Saving model at epoch:", epoch)
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch)

        # 更新 TensorBoard 日志
        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        # 记录训练和验证统计数据
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     **{f"test_{k}": v for k, v in test_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}

        # 如果是主进程且指定了输出目录，保存日志
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 打印总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

