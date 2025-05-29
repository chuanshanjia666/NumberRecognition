import os
import numpy as np
import tensorrt as trt
from PIL import Image
import common  # 导入 common 模块

# TensorRT Logger
TRT_LOGGER = trt.Logger()

# 加载 TensorRT 引擎
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 预处理测试图片
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # 确保转换为灰度图
    image = np.array(image).astype(np.float32) / 255.0  # 归一化
    image = (image - 0.1307) / 0.3081  # 标准化
    image = np.expand_dims(image, axis=0)  # 添加 batch 维度
    image = np.expand_dims(image, axis=0)  # 最终形状：(1, 1, 28, 28)
    return image

# 主函数
def main():
    # TensorRT 引擎文件路径
    engine_file_path = "mnist_cnn.trt"  # 替换为你的引擎文件路径
    test_image_path = "test.png"  # 替换为你的测试图片路径

    # 加载 TensorRT 引擎
    engine = load_engine(engine_file_path)
    if engine is None:
        print("Failed to load TensorRT engine.")
        return
    print("TensorRT engine loaded successfully.")

    # 加载并预处理测试图片
    input_data = preprocess_image(test_image_path)
    print(f"Input data shape: {input_data.shape}")

    # 分配输入/输出缓冲区
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    # 将输入数据复制到主机缓冲区
    np.copyto(inputs[0].host, input_data.ravel())

    # 创建执行上下文
    context = engine.create_execution_context()

    # 执行推理
    output = common.do_inference(
        context,
        engine=engine,  # 传入 engine 参数
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream
    )
    print("Output:", output)

    # 解析输出结果
    predicted_class = np.argmax(output[0])
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()