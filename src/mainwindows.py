import sys
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5 import uic
from tensorrt_recognizer import TensorRTRecognizer
import config

class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.brush_size = config.BRUSH_SIZE
        self.brush_color = Qt.white
        self.last_point = QPoint()
        
        # 创建画布
        canvas_size = config.CANVAS_SIZE
        self.pixmap = QPixmap(canvas_size[0], canvas_size[1])
        self.pixmap.fill(Qt.black)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(self.brush_color, self.brush_size, 
                             Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            
    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawPixmap(0, 0, self.pixmap)
        
    def clear_canvas(self):
        """清除画布"""
        self.pixmap.fill(Qt.black)
        self.update()
        
    def get_image_data(self):
        """获取画布图像数据"""
        return self.pixmap

class HandwritingRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 加载UI文件
        uic.loadUi('mainwindows.ui', self)
        
        # 初始化TensorRT识别器
        self.init_recognizer()
        
        # 初始化手写区域
        self.setup_drawing_area()
        
        # 连接按钮信号
        self.clearButton.clicked.connect(self.clear_drawing)
        self.recognizeButton.clicked.connect(self.recognize_character)
        
        # 设置状态栏
        self.update_status_message()
        
    def init_recognizer(self):
        """初始化TensorRT识别器"""
        try:
            engine_path = config.TENSORRT_ENGINE_PATH
            self.recognizer = TensorRTRecognizer(engine_path)
            
            if not self.recognizer.is_ready():
                self.recognizer = None
                print("Warning: TensorRT recognizer not available, using simulation mode")
        except Exception as e:
            print(f"Error initializing TensorRT recognizer: {e}")
            self.recognizer = None
            
    def update_status_message(self):
        """更新状态栏消息"""
        if self.recognizer and self.recognizer.is_ready():
            self.statusbar.showMessage("TensorRT模式：请在左侧区域手写数字 (0-9)")
        else:
            self.statusbar.showMessage("模拟模式：请在左侧区域手写字符")
        
    def setup_drawing_area(self):
        """设置手写区域"""
        # 移除原有的widget
        self.drawingWidget.setParent(None)
        
        # 创建新的绘图widget
        self.drawing_widget = DrawingWidget()
        
        # 将新的绘图widget添加到布局中
        # 获取水平布局
        horizontal_layout = self.centralwidget.findChild(self.centralwidget.__class__, 'horizontalLayout')
        if hasattr(self, 'horizontalLayout'):
            self.horizontalLayout.insertWidget(0, self.drawing_widget)
        else:
            # 如果找不到布局，直接替换
            self.drawing_widget.setParent(self.centralwidget)
            self.drawing_widget.setGeometry(50, 80, 400, 400)
        
    def clear_drawing(self):
        """清除手写内容"""
        self.drawing_widget.clear_canvas()
        self.characterLabel.setText("-")
        self.confidenceLabel.setText("置信度：-")
        self.statusbar.showMessage("画布已清除，请重新手写")
        
    def pixmap_to_numpy(self, pixmap):
        """将QPixmap转换为numpy数组"""
        try:
            # 转换为QImage
            qimage = pixmap.toImage()
            
            # 转换为RGB格式
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
            
            # 获取图像数据
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            
            # 转换为numpy数组
            arr = np.array(ptr).reshape(height, width, 3)
            
            # 转换为灰度图 (取R通道，因为画布是黑白的)
            gray_arr = arr[:, :, 0]
            
            return gray_arr
            
        except Exception as e:
            print(f"Error converting pixmap to numpy: {e}")
            return None
        
    def recognize_character(self):
        """识别手写字符"""
        # 获取图像数据
        image_pixmap = self.drawing_widget.get_image_data()
        
        # 检查是否有手写内容
        if self.is_canvas_empty():
            self.statusbar.showMessage("请先手写字符再进行识别")
            return
            
        # 使用TensorRT识别或模拟识别
        if self.recognizer and self.recognizer.is_ready():
            self.tensorrt_recognition(image_pixmap)
        else:
            self.simulate_recognition()
            
    def tensorrt_recognition(self, image_pixmap):
        """使用TensorRT进行真实识别"""
        try:
            # 转换QPixmap为numpy数组
            image_array = self.pixmap_to_numpy(image_pixmap)
            
            if image_array is None:
                self.statusbar.showMessage("图像处理失败")
                return
                
            # 执行识别
            predicted_class, confidence, probabilities = self.recognizer.recognize(image_array)
            
            if predicted_class is not None:
                # 显示识别结果
                self.characterLabel.setText(str(predicted_class))
                self.confidenceLabel.setText(f"置信度：{confidence:.1f}%")
                self.statusbar.showMessage(f"识别完成：{predicted_class} (置信度: {confidence:.1f}%)")
                
                # 在控制台输出详细信息
                print(f"Recognition result: {predicted_class}")
                print(f"Confidence: {confidence:.2f}%")
                if probabilities is not None:
                    print("All probabilities:")
                    for i, prob in enumerate(probabilities):
                        print(f"  {i}: {prob*100:.2f}%")
            else:
                self.statusbar.showMessage("识别失败，请重试")
                
        except Exception as e:
            print(f"TensorRT recognition error: {e}")
            self.statusbar.showMessage("识别过程出现错误")
            
    def simulate_recognition(self):
        """模拟字符识别过程（备用模式）"""
        # 使用数字类别进行模拟
        recognized_char = random.choice(config.DIGIT_CLASSES)
        confidence = random.uniform(75.0, 98.5)
        
        # 显示识别结果
        self.characterLabel.setText(recognized_char)
        self.confidenceLabel.setText(f"置信度：{confidence:.1f}%")
        self.statusbar.showMessage(f"模拟识别：{recognized_char} (置信度: {confidence:.1f}%)")
        
    def is_canvas_empty(self):
        """检查画布是否为空"""
        # 简单的检查方法：创建一个全黑的pixmap比较
        empty_pixmap = QPixmap(400, 400)
        empty_pixmap.fill(Qt.black)
        return self.drawing_widget.pixmap.toImage() == empty_pixmap.toImage()
        
    def simulate_recognition(self):
        """模拟字符识别过程"""
        # 模拟识别结果
        characters = ['A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', '5', 
                     '你', '好', '中', '文', '字']
        recognized_char = random.choice(characters)
        confidence = random.uniform(75.0, 98.5)
        
        return recognized_char, confidence

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("手写字符识别")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = HandwritingRecognition()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()