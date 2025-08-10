import sys
import numpy as np
import mss
import torch
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Overlay")
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowTransparentForInput |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.screen = QtWidgets.QApplication.primaryScreen()
        size = self.screen.size()
        self.setGeometry(0, 0, size.width(), size.height())
        self.detections = []

    def setDetections(self, detections):
        self.detections = detections
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 200), 3))
        painter.setFont(QtGui.QFont('Arial', 12))

        for (x1, y1, x2, y2, conf) in self.detections:
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, y1 - 10, f'Person {conf:.2f}')


class DetectionWorker(QtCore.QThread):
    detectionsReady = QtCore.pyqtSignal(list)

    def __init__(self, model, monitor, screen_width, screen_height):
        super().__init__()
        self.model = model
        self.monitor = monitor
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.running = True

    def run(self):
        sct = mss.mss()  # create MSS instance here inside thread
        while self.running:
            img = np.array(sct.grab(self.monitor))
            frame = img[:, :, :3]  # BGRA to BGR

            # Resize smaller for faster inference
            small_frame = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))

            # Convert BGR to RGB
            img_rgb = small_frame[:, :, ::-1]

            # Run inference
            results = self.model.predict(img_rgb, device=self.device, imgsz=320, conf=0.25)

            scale_x = self.screen_width / small_frame.shape[1]
            scale_y = self.screen_height / small_frame.shape[0]

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls.cpu())
                    conf = float(box.conf.cpu())
                    if cls == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        detections.append((x1, y1, x2, y2, conf))

            self.detectionsReady.emit(detections)


class MainApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.screen = QtWidgets.QApplication.primaryScreen()
        size = self.screen.size()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt')
        self.model.to(device)

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

        self.overlay = OverlayWindow()
        self.overlay.show()

        self.worker = DetectionWorker(self.model, self.monitor, size.width(), size.height())
        self.worker.detectionsReady.connect(self.overlay.setDetections)
        self.worker.start()

    def run(self):
        sys.exit(self.app.exec_())

    def stop(self):
        self.worker.running = False
        self.worker.wait()


if __name__ == "__main__":
    mainapp = MainApp()
    try:
        mainapp.run()
    except KeyboardInterrupt:
        mainapp.stop()
