import os
import numpy as np
import tensorrt as trt
from PIL import Image
import common  # 导入 common 模块
import config  # 导入配置文件

class TensorRTRecognizer:
    """TensorRT手写数字识别器"""
    
    def __init__(self, engine_file_path):
        """
        初始化TensorRT识别器
        
        Args:
            engine_file_path (str): TensorRT引擎文件路径
        """
        self.engine_file_path = engine_file_path
        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        
        # TensorRT Logger
        self.trt_logger = trt.Logger()
        
        # 初始化引擎
        self._load_engine()
        
    def _load_engine(self):
        """加载TensorRT引擎"""
        try:
            if not os.path.exists(self.engine_file_path):
                raise FileNotFoundError(f"Engine file not found: {self.engine_file_path}")
                
            with open(self.engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            print("TensorRT engine loaded successfully.")
            
            # 分配缓冲区
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
            
            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            self.engine = None
            
    def is_ready(self):
        """检查识别器是否准备就绪"""
        return self.engine is not None and self.context is not None
        
    def preprocess_image(self, image_array):
        """
        预处理图像数据
        
        Args:
            image_array (numpy.ndarray): 输入图像数组，形状为(height, width)或(height, width, channels)
            
        Returns:
            numpy.ndarray: 预处理后的图像数据，形状为(1, 1, 28, 28)
        """
        try:
            # 确保输入是numpy数组
            if not isinstance(image_array, np.ndarray):
                image_array = np.array(image_array)
            
            # 如果是彩色图像，转换为灰度图
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:  # RGB
                    image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
                elif image_array.shape[2] == 4:  # RGBA
                    image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # 调整大小到28x28
            target_size = config.IMAGE_SIZE
            image_pil = Image.fromarray(image_array.astype(np.uint8))
            image_pil = image_pil.resize(target_size, Image.LANCZOS)
            image_array = np.array(image_pil)
            
            # 归一化到[0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # 标准化 (MNIST数据集的均值和标准差)
            image_array = (image_array - config.MNIST_MEAN) / config.MNIST_STD
            
            # 调整形状为(1, 1, 28, 28)
            image_array = np.expand_dims(image_array, axis=0)  # 添加channel维度
            image_array = np.expand_dims(image_array, axis=0)  # 添加batch维度
            
            return image_array
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
            
    def recognize(self, image_array):
        """
        识别手写数字
        
        Args:
            image_array (numpy.ndarray): 输入图像数组
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
                predicted_class (int): 预测的类别 (0-9)
                confidence (float): 置信度 (0-100)
                all_probabilities (numpy.ndarray): 所有类别的概率分布
        """
        if not self.is_ready():
            print("TensorRT engine is not ready")
            return None, 0.0, None
            
        try:
            # 预处理图像
            input_data = self.preprocess_image(image_array)
            if input_data is None:
                return None, 0.0, None
                
            print(f"Input data shape: {input_data.shape}")
            
            # 将输入数据复制到主机缓冲区
            np.copyto(self.inputs[0].host, input_data.ravel())
            
            # 执行推理
            output = common.do_inference(
                self.context,
                engine=self.engine,
                bindings=self.bindings,
                inputs=self.inputs,
                outputs=self.outputs,
                stream=self.stream
            )
            
            # 解析输出结果
            probabilities = np.array(output[0])
            
            # 应用softmax获得概率分布
            exp_scores = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_scores / np.sum(exp_scores)
            
            # 获取预测类别和置信度
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class] * 100
            
            print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
            
            return int(predicted_class), float(confidence), probabilities
            
        except Exception as e:
            print(f"Error during recognition: {e}")
            return None, 0.0, None
            
    def recognize_from_file(self, image_path):
        """
        从文件识别手写数字
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        try:
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)
            return self.recognize(image_array)
        except Exception as e:
            print(f"Error loading image from file: {e}")
            return None, 0.0, None
            
    def release_resources(self):
        """释放所有分配的资源"""
        if self.inputs is not None and self.outputs is not None and self.stream is not None:
            common.free_buffers(self.inputs, self.outputs, self.stream)
            self.inputs = None
            self.outputs = None
            self.bindings = None
            self.stream = None
            
        if self.context is not None:
            del self.context
            self.context = None
            
        if self.engine is not None:
            del self.engine
            self.engine = None
            
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.release_resources()