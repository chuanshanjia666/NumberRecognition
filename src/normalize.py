import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


def get_mean_var_from_dataset_rgb(
    dataset: Dataset,
    sample_ratio: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集中抽样并计算均值和方差

    Args:
        dataset: 数据集对象（torch.utils.data.Dataset）
        sample_ratio: 抽样比例（默认 0.1，即 10% 的数据）
        batch_size: 批次大小（默认 64）
        num_workers: 数据加载时使用的线程数（默认 0）

    Returns:
        mean: 样本的均值（按通道计算），形状为 (C,)，C 是通道数
        var: 样本的方差（按通道计算），形状为 (C,)，C 是通道数
    """
    # 计算抽样数量
    sample_size: int = int(len(dataset) * sample_ratio)
    print(f"抽样数量: {sample_size} 张图片")

    # 随机抽样
    subset: Dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:sample_size])

    # 创建 DataLoader
    loader: DataLoader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers)

    # 初始化变量
    mean: torch.Tensor = torch.zeros(3)  # 假设是 3 通道（RGB）
    var: torch.Tensor = torch.zeros(3)
    total_pixels: int = 0

    # 计算均值和方差
    for images, _ in loader:
        B: int
        C: int
        H: int
        W: int
        B, C, H, W = images.shape
        num_pixels: int = B * H * W
        total_pixels += num_pixels

        # 计算每个通道的均值和方差
        for c in range(C):
            channel_data: torch.Tensor = images[:, c, :, :]  # 当前通道的所有像素
            mean[c] += channel_data.sum()
            var[c] += (channel_data ** 2).sum()

    # 归一化
    mean /= total_pixels
    var = var / total_pixels - mean ** 2

    return mean, var

def get_mean_var_from_dataset_grey(
    dataset: Dataset,
    sample_ratio: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从灰度图像数据集中抽样并计算均值和方差

    Args:
        dataset: 数据集对象（torch.utils.data.Dataset）
        sample_ratio: 抽样比例（默认 0.1，即 10% 的数据）
        batch_size: 批次大小（默认 64）
        num_workers: 数据加载时使用的线程数（默认 0）

    Returns:
        mean: 样本的均值，形状为 (1,)
        var: 样本的方差，形状为 (1,)
    """
    # 计算抽样数量
    sample_size: int = int(len(dataset) * sample_ratio)
    print(f"抽样数量: {sample_size} 张图片")

    # 随机抽样
    subset: Dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:sample_size])

    # 创建 DataLoader
    loader: DataLoader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers)

    # 初始化变量
    mean: torch.Tensor = torch.zeros(1)  # 灰度图只有一个通道
    var: torch.Tensor = torch.zeros(1)
    total_pixels: int = 0

    # 计算均值和方差
    for images, _ in loader:
        B, C, H, W = images.shape
        num_pixels: int = B * H * W
        total_pixels += num_pixels

        # 计算均值和方差（只有单通道）
        channel_data: torch.Tensor = images[:, 0, :, :]  # 灰度图只有一个通道
        mean += channel_data.sum()
        var += (channel_data ** 2).sum()

    # 归一化
    mean /= total_pixels
    var = var / total_pixels - mean ** 2

    return mean, var