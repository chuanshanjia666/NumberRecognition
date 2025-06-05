import sys
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
        self.pixmap.fill(Qt.black)
        self.update()
        
    def get_image_data(self):
        return self.pixmap

class HandwritingRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        
        uic.loadUi('mainwindows.ui', self)
        
        if not self.init_recognizer():
            error_msg = "无法加载TensorRT引擎文件，请确保模型文件存在且路径正确。"
            QMessageBox.critical(self, "初始化失败", error_msg)
            sys.exit(1)
            
        self.setup_drawing_area()
        
        self.clearButton.clicked.connect(self.clear_drawing)
        self.recognizeButton.clicked.connect(self.recognize_character)
        
        self.statusbar.showMessage("请在左侧区域手写数字 (0-9)")
        
    def init_recognizer(self):
        try:
            engine_path = config.TENSORRT_ENGINE_PATH
            self.recognizer = TensorRTRecognizer(engine_path)
            
            if not self.recognizer.is_ready():
                print(f"Error: TensorRT引擎不可用，路径: {engine_path}")
                return False
            return True
            
        except Exception as e:
            print(f"Error initializing TensorRT recognizer: {e}")
            return False
            
    def setup_drawing_area(self):
        self.drawingWidget.setParent(None)
        
        self.drawing_widget = DrawingWidget()
        
        horizontal_layout = self.centralwidget.findChild(self.centralwidget.__class__, 'horizontalLayout')
        if hasattr(self, 'horizontalLayout'):
            self.horizontalLayout.insertWidget(0, self.drawing_widget)
        else:
            self.drawing_widget.setParent(self.centralwidget)
            self.drawing_widget.setGeometry(50, 80, 400, 400)
        
    def clear_drawing(self):
        self.drawing_widget.clear_canvas()
        self.characterLabel.setText("-")
        self.confidenceLabel.setText("置信度：-")
        self.statusbar.showMessage("画布已清除，请重新手写")
        
    def pixmap_to_numpy(self, pixmap):
        try:
            qimage = pixmap.toImage()
            
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
            
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            
            arr = np.array(ptr).reshape(height, width, 3)
            
            gray_arr = arr[:, :, 0]
            
            return gray_arr
            
        except Exception as e:
            print(f"Error converting pixmap to numpy: {e}")
            return None
        
    def recognize_character(self):
        image_pixmap = self.drawing_widget.get_image_data()
        
        if self.is_canvas_empty():
            self.statusbar.showMessage("请先手写字符再进行识别")
            return
            
        self.tensorrt_recognition(image_pixmap)
            
    def tensorrt_recognition(self, image_pixmap):
        try:
            image_array = self.pixmap_to_numpy(image_pixmap)
            
            if image_array is None:
                self.statusbar.showMessage("图像处理失败")
                return
                
            predicted_class, confidence, probabilities = self.recognizer.recognize(image_array)
            
            if predicted_class is not None:
                self.characterLabel.setText(str(predicted_class))
                self.confidenceLabel.setText(f"置信度：{confidence:.1f}%")
                self.statusbar.showMessage(f"识别完成：{predicted_class} (置信度: {confidence:.1f}%)")
                
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
            
    def is_canvas_empty(self):
        empty_pixmap = QPixmap(400, 400)
        empty_pixmap.fill(Qt.black)
        return self.drawing_widget.pixmap.toImage() == empty_pixmap.toImage()

def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("手写数字识别")
    app.setApplicationVersion("1.0")
    
    window = HandwritingRecognition()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()    