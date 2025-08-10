import sys
import numpy as np
import mss
import torch
import cv2
import pyautogui
import time
import keyboard
from PyQt5 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO
import threading

pyautogui.FAILSAFE = False  # Disable failsafe for uninterrupted mouse movement

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
        self.screen_width = size.width()
        self.screen_height = size.height()

    def setDetections(self, detections):
        self.detections = detections
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        pen_box = QtGui.QPen(QtGui.QColor(0, 255, 0, 200), 3)
        pen_line = QtGui.QPen(QtGui.QColor(255, 0, 0, 180), 2)  # tracer line red
        font = QtGui.QFont('Arial', 12)
        painter.setFont(font)

        center_x = self.screen_width // 2
        bottom_y = self.screen_height

        for (x1, y1, x2, y2, conf) in self.detections:
            painter.setPen(pen_box)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, y1 - 10, f'Person {conf:.2f}')

            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            painter.setPen(pen_line)
            painter.drawLine(center_x, bottom_y, box_center_x, box_center_y)

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
        sct = mss.mss()
        while self.running:
            img = np.array(sct.grab(self.monitor))
            frame = img[:, :, :3]  # BGR

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.model.predict(
                img_rgb,
                device=self.device,
                imgsz=640,
                conf=0.35,
                verbose=False,
                classes=[0]
            )

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf.cpu())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    detections.append((x1, y1, x2, y2, conf))

            self.detectionsReady.emit(detections)
            time.sleep(0.01)

class MainApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.screen = QtWidgets.QApplication.primaryScreen()
        size = self.screen.size()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8s.pt')
        self.model.to(device)

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

        self.overlay = OverlayWindow()
        self.overlay.show()

        self.worker = DetectionWorker(self.model, self.monitor, size.width(), size.height())
        self.worker.detectionsReady.connect(self.overlay.setDetections)
        self.worker.detectionsReady.connect(self.aim_at_target)
        self.worker.start()

        self.aim_enabled = False

        self.running = True
        threading.Thread(target=self.listen_for_f6_toggle, daemon=True).start()

    def aim_at_target(self, detections):
        if not self.aim_enabled or not detections:
            return

        target = max(detections, key=lambda d: d[4])
        x1, y1, x2, y2, conf = target

        target_x = (x1 + x2) // 2
        target_y = y1 + int((y2 - y1) * 0.2)

        # Move mouse instantly to target position (no smoothing)
        pyautogui.moveTo(target_x, target_y, duration=0)

    def listen_for_f6_toggle(self):
        while self.running:
            if keyboard.is_pressed('f6'):
                self.aim_enabled = not self.aim_enabled
                print(f"Aim assist toggled {'ON' if self.aim_enabled else 'OFF'}")
                time.sleep(0.5)
            time.sleep(0.05)

    def run(self):
        sys.exit(self.app.exec_())

    def stop(self):
        self.running = False
        self.worker.running = False
        self.worker.wait()

if __name__ == "__main__":
    mainapp = MainApp()
    try:
        mainapp.run()
    except KeyboardInterrupt:
        mainapp.stop()
