# NumberRecognition
云计算大作业——基于MNIST数据集的手写识别

## 1.数据爬取+预处理+模型训练

### 数据爬取

相关文件：`get_data_and_train.py`

从网络上下载MINST数据集，从这三个网站上下载

- https://mirrors.ustc.edu.cn/mnist/
- https://ossci-datasets.s3.amazonaws.com/mnist/
- https://storage.googleapis.com/cvdf-datasets/mnist/

下载后解压

### 预处理
使用解压后的数据，一项一项加载数据并添加标签
对数据进行归一化，将0-255的像素数据放缩到0,1
对数据进行归一化，将数据放缩为N(0,1)

### 模型训练
使用pytorch对模型进行训练

```python
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
```



 使用**负对数似然损失**作为损失函数，也可以认为是**交叉熵损失**

使用**Adam 优化器**作为优化函数

运行结果

![pytorch](.\image\pytorch.png)

## 2.模型转换

相关文件：`to_tensorrt.py`

环境:CUDA 12.4 tensorRT 10

参考了官方提供的例程

核心代码：

```python
def get_engine(onnx_file_path, engine_file_path=""):

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            0
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 1 << 28
            )  # 256MiB
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                        onnx_file_path
                    )
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            network.get_input(0).shape = [1, 1, 28, 28]
            print("Completed parsing of ONNX file")
            print(
                "Building an engine from file {}; this may take a while...".format(
                    onnx_file_path
                )
            )
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
```

其中**network.get_input(0).shape = [1, 1, 28, 28]**需要根据模型输入自行调整



## 3.可视化部分

相关文件:

`common.py` `common_runtime.py` NVIDIA官方提供的文件，实现了模型推理和一些工具函数

`tensorrt_recognizer.py` 实现了数据到输入数据格式的转换和模型推理

`mainwindows.ui` `mainwindows.py`实现了结果可视化：用户交互

效果图

![effect](.\image\effect.png)
