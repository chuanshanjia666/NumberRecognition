import os
import numpy as np
import tensorrt as trt
from PIL import Image
import common
import config

class TensorRTRecognizer:
    def __init__(self, engine_file_path):
        self.engine_file_path = engine_file_path
        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.trt_logger = trt.Logger()
        self._load_engine()
        
    def _load_engine(self):
        try:
            if not os.path.exists(self.engine_file_path):
                raise FileNotFoundError(f"Engine file not found: {self.engine_file_path}")
                
            with open(self.engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            print("TensorRT engine loaded successfully.")
            
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
            
            self.context = self.engine.create_execution_context()
            
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            self.engine = None
            
    def is_ready(self):
        return self.engine is not None and self.context is not None
        
    def preprocess_image(self, image_array):
        try:
            if not isinstance(image_array, np.ndarray):
                image_array = np.array(image_array)
            
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
                elif image_array.shape[2] == 4:
                    image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            target_size = config.IMAGE_SIZE
            image_pil = Image.fromarray(image_array.astype(np.uint8))
            image_pil = image_pil.resize(target_size, Image.LANCZOS)
            image_array = np.array(image_pil)
            
            image_array = image_array.astype(np.float32) / 255.0
            
            image_array = (image_array - config.MNIST_MEAN) / config.MNIST_STD
            
            image_array = np.expand_dims(image_array, axis=0)
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
            
    def recognize(self, image_array):
        if not self.is_ready():
            print("TensorRT engine is not ready")
            return None, 0.0, None
            
        try:
            input_data = self.preprocess_image(image_array)
            if input_data is None:
                return None, 0.0, None
                
            print(f"Input data shape: {input_data.shape}")
            
            np.copyto(self.inputs[0].host, input_data.ravel())
            
            output = common.do_inference(
                self.context,
                engine=self.engine,
                bindings=self.bindings,
                inputs=self.inputs,
                outputs=self.outputs,
                stream=self.stream
            )
            
            probabilities = np.array(output[0])
            
            exp_scores = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_scores / np.sum(exp_scores)
            
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class] * 100
            
            print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
            
            return int(predicted_class), float(confidence), probabilities
            
        except Exception as e:
            print(f"Error during recognition: {e}")
            return None, 0.0, None
            
    def recognize_from_file(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)
            return self.recognize(image_array)
        except Exception as e:
            print(f"Error loading image from file: {e}")
            return None, 0.0, None
            
    def release_resources(self):
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
        self.release_resources()