# 配置文件

# TensorRT引擎文件路径
TENSORRT_ENGINE_PATH = "mnist_cnn.trt"

# 图像预处理参数
IMAGE_SIZE = (28, 28)  # 输入图像大小
MNIST_MEAN = 0.1307    # MNIST数据集均值
MNIST_STD = 0.3081     # MNIST数据集标准差

# UI设置
CANVAS_SIZE = (400, 400)  # 画布大小
BRUSH_SIZE = 18            # 画笔大小
BRUSH_COLOR = "white"     # 画笔颜色
BACKGROUND_COLOR = "black"  # 背景颜色

# 数字类别标签
DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']