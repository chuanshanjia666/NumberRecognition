# 添加导出 ONNX 的代码
import torch
import onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)  # 计算公式: 64*(28-2*2)^2 = 9216
        self.fc2 = nn.Linear(128, 10)    # 输出10类

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [1,28,28] -> [32,26,26]
        x = F.relu(self.conv2(x))   # -> [64,24,24]
        x = F.max_pool2d(x, 2)      # -> [64,12,12]
        x = self.dropout(x)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # 负对数似然损失
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")

# MNIST 数据集
# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(1, 6):  # 训练5轮
    train(epoch)

# 测试模型
test()

# 保存模型
torch.save(model.state_dict(), "mnist_cnn.pt")
torch.save(model, "mnist_cnn_full.pth")

# 导出 ONNX 模型
dummy_input = torch.randn(1, 1, 28, 28).to(device)  # 创建一个与模型输入维度相同的示例输入
onnx_path = "mnist_cnn.onnx"
torch.onnx.export(
    model,                        # 要导出的模型
    dummy_input,                  # 示例输入
    onnx_path,                    # ONNX 文件路径
    input_names=["input"],        # 输入节点名称
    output_names=["output"],      # 输出节点名称
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # 支持动态 batch size
    opset_version=11              # ONNX 算子集版本
)

print(f"模型已成功导出为 ONNX 格式: {onnx_path}")