import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 定义模型结构（必须与训练时一致）
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
        return F.log_softmax(x, dim=1)  # 输出 LogSoftmax 结果

# 加载训练好的模型
def load_model(model_path):
    # 初始化模型
    model = CNN()
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))  # 加载到 CPU
    model.eval()  # 设置为评估模式
    return model

# 预处理测试图片
def preprocess_image(image_path):
    # 加载图片并转换为灰度
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    # 定义预处理流程
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整大小
        transforms.ToTensor(),        # 转为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])
    # 进行预处理并添加 batch 维度
    image = transform(image).unsqueeze(0)  # 添加 batch 维度，形状：(1, 1, 28, 28)
    return image

# 推理
def infer(model, input_data):
    with torch.no_grad():  # 关闭梯度计算
        output = model(input_data)  # 前向传播
        _, predicted_class = torch.max(output, 1)  # 获取预测类别
        return output, predicted_class.item()

# 主函数
def main():
    # 模型权重文件路径
    model_path = "mnist_cnn.pt"  # 替换为你的模型权重文件路径
    # 测试图片路径
    test_image_path = "test.png"  # 替换为你的测试图片路径

    # 加载模型
    model = load_model(model_path)
    print("Model loaded successfully.")

    # 预处理测试图片
    input_data = preprocess_image(test_image_path)
    print(f"Input data shape: {input_data.shape}")

    # 执行推理
    output, predicted_class = infer(model, input_data)
    print("Model output (LogSoftmax):", output)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()