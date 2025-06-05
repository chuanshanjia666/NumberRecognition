import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import struct
import requests
import gzip
import shutil
import hashlib

class CustomMNISTDataset(Dataset):
    """自定义 MNIST 数据集加载器，支持自动下载和解压"""
    
    # MNIST文件的预期大小（字节）
    EXPECTED_FILE_SIZES = {
        'train-images-idx3-ubyte.gz': 9912422,
        'train-labels-idx1-ubyte.gz': 28881,
        't10k-images-idx3-ubyte.gz': 1648877,
        't10k-labels-idx1-ubyte.gz': 4542
    }
    
    def __init__(self, images_path, labels_path=None, transform=None, download=True):
        """
        初始化数据集
        
        Args:
            images_path (str): 图像文件路径(.gz格式)
            labels_path (str): 标签文件路径(.gz格式)，测试时可以设为None
            transform (callable): 数据预处理转换
            download (bool): 如果文件不存在是否自动下载
        """
        self.transform = transform
        
        # 自动下载和解压文件
        if download:
            self._download_and_extract(images_path)
            if labels_path:
                self._download_and_extract(labels_path)
        
        # 加载数据
        self.images = self._load_images(images_path.replace('.gz', ''))
        self.labels = self._load_labels(labels_path.replace('.gz', '')) if labels_path else None
        
    def _download_and_extract(self, file_path):
        """下载并解压MNIST文件 - 增强的错误处理"""
        # 使用中国科技大学镜像源
        mirror_urls = [
            "https://mirrors.ustc.edu.cn/mnist/",  # 中国科技大学镜像
            "https://ossci-datasets.s3.amazonaws.com/mnist/",  # AWS镜像(备用)
            "https://storage.googleapis.com/cvdf-datasets/mnist/",  # Google镜像(备用)
        ]
        
        file_name = os.path.basename(file_path)
        raw_file = file_path.replace('.gz', '')
        
        # 如果文件已存在，检查其大小是否符合预期
        if os.path.exists(raw_file):
            expected_size = self._get_expected_size(file_name.replace('.gz', ''))
            actual_size = os.path.getsize(raw_file)
            
            if expected_size and actual_size == expected_size:
                print(f"文件 {file_name} 已存在且大小正确，跳过下载")
                return
            else:
                print(f"文件 {file_name} 存在但大小不匹配，重新下载")
                os.remove(raw_file)
        
        # 下载.gz文件（最多尝试5次）
        max_retries = 5
        for attempt in range(max_retries):
            # 删除可能存在的不完整文件
            if os.path.exists(file_path):
                os.remove(file_path)
            
            print(f"尝试下载 {file_name} (尝试 {attempt+1}/{max_retries})...")
            
            success = False
            for base_url in mirror_urls:
                try:
                    url = base_url + file_name
                    print(f"Trying URL: {url}")
                    
                    # 发送HEAD请求获取文件大小
                    head_response = requests.head(url, timeout=30)
                    if head_response.status_code != 200:
                        print(f"HEAD请求失败，状态码: {head_response.status_code}")
                        continue
                    
                    expected_size = int(head_response.headers.get('Content-Length', 0))
                    if expected_size == 0:
                        print("无法获取文件大小信息")
                        continue
                    
                    # 开始下载文件，带进度显示
                    print(f"开始下载 {file_name} ({expected_size/1024/1024:.2f} MB)")
                    response = requests.get(url, stream=True, timeout=60)  # 增加超时时间到60秒
                    
                    if response.status_code != 200:
                        print(f"下载请求失败，状态码: {response.status_code}")
                        continue
                    
                    # 下载文件并显示进度
                    with open(file_path, 'wb') as f:
                        downloaded_size = 0
                        chunk_size = 1024 * 1024  # 1MB
                        
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                progress = downloaded_size / expected_size * 100
                                print(f"\r下载进度: {progress:.2f}% ({downloaded_size/1024/1024:.2f}/{expected_size/1024/1024:.2f} MB)", end='')
                        
                        print()  # 换行
                    
                    # 验证下载的文件大小
                    actual_size = os.path.getsize(file_path)
                    if actual_size == expected_size:
                        print(f"文件 {file_name} 下载完成，大小验证通过")
                        success = True
                        break
                    else:
                        print(f"文件 {file_name} 大小不匹配: 预期 {expected_size} 字节，实际 {actual_size} 字节")
                        print("删除不完整文件并尝试其他镜像...")
                
                except Exception as e:
                    print(f"下载过程中出错: {e}")
                    continue
            
            if success:
                break
        
        if not success:
            print(f"下载 {file_name} 失败，尝试了所有镜像源和 {max_retries} 次重试")
            print("建议使用torchvision.datasets.MNIST代替自定义下载。")
            raise RuntimeError(f"无法下载文件 {file_name}")
        
        # 解压.gz文件
        print(f"开始解压 {file_name}...")
        try:
            with gzip.open(file_path, 'rb') as f_in:
                with open(raw_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"解压 {file_name} 完成")
        except Exception as e:
            print(f"解压失败 {file_name}: {e}")
            # 删除可能不完整的解压文件
            if os.path.exists(raw_file):
                os.remove(raw_file)
            raise
    
    def _get_expected_size(self, file_name):
        """获取文件的预期大小"""
        if file_name in self.EXPECTED_FILE_SIZES:
            return self.EXPECTED_FILE_SIZES[file_name]
        # 根据文件类型估算大小
        if "images" in file_name:
            return 60000 * 28 * 28 + 16  # 训练集图像大小
        elif "labels" in file_name:
            return 60000 + 8  # 训练集标签大小
        return None
    
    def _load_images(self, file_path):
        """加载图像数据"""
        print(f"加载图像数据: {file_path}")
        with open(file_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            print(f"读取到 {num} 张 {rows}x{cols} 的图像")
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images
    
    def _load_labels(self, file_path):
        """加载标签数据"""
        print(f"加载标签数据: {file_path}")
        with open(file_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            print(f"读取到 {num} 个标签")
            labels = np.fromfile(f, dtype=np.uint8)
        return labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        if self.labels is not None:
            return image, self.labels[idx]
        return image


def load_mnist_with_fallback():
    """
    尝试加载MNIST数据集，如果自定义加载失败则使用torchvision
    """
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        # 尝试使用自定义数据集加载器
        print("尝试使用自定义数据集加载器...")
        
        # 文件路径配置
        data_dir = "mnist_data"
        os.makedirs(data_dir, exist_ok=True)

        train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

        # 加载自定义数据集
        train_dataset = CustomMNISTDataset(
            images_path=train_images_path,
            labels_path=train_labels_path,
            transform=transform,
            download=True
        )

        test_dataset = CustomMNISTDataset(
            images_path=test_images_path,
            labels_path=test_labels_path,
            transform=transform,
            download=True
        )
        
        print("自定义数据集加载成功！")
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"自定义数据集加载失败: {e}")
        print("回退到使用 torchvision.datasets.MNIST...")
        
        # 使用torchvision的MNIST数据集作为备选方案
        train_dataset = datasets.MNIST(
            root='./torchvision_data',
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root='./torchvision_data',
            train=False,
            download=True,
            transform=transform
        )
        
        print("使用torchvision数据集加载成功！")
        return train_dataset, test_dataset


# CNN模型定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 训练和测试函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")
    return acc


def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集（带备选方案）
    train_dataset, test_dataset = load_mnist_with_fallback()
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("Starting training...")
    best_acc = 0
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            # 保存最佳模型
            torch.save(model.state_dict(), "mnist_cnn_custom_best.pt")
    
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    
    # 保存最终模型
    torch.save(model.state_dict(), "mnist_cnn_custom_final.pt")
    
    # 导出ONNX模型
    try:
        print("Exporting ONNX model...")
        model.eval()
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        onnx_path = "mnist_cnn_custom.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        print(f"模型已成功导出为 ONNX 格式: {onnx_path}")
    except Exception as e:
        print(f"ONNX导出失败: {e}")


if __name__ == "__main__":
    main()