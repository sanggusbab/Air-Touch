import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import pyautogui
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from click import HandClickDetector
from gaze import GazeEstimator

class TransparentWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_x = 0
        self.current_y = 0

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setGeometry(0, 0, 1080, 1920)
        self.show()

        self.waves = []
        self.max_radius = 100
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_effect)
        self.timer.start(50)

    def update_effect(self):
        new_waves = []
        for (x, y, radius) in self.waves:
            if radius < self.max_radius:
                new_waves.append((x, y, radius + 5))
        self.waves = new_waves
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        for (x, y, radius) in self.waves:
            alpha = max(0, 127 - int(127 * (radius / self.max_radius)))
            gradient_color = QColor(0, 255, 255, alpha)
            pen = QPen(gradient_color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(x - radius, y - radius, 2 * radius, 2 * radius)

    def setGazeCoordinates(self, x, y):
        self.waves.append((x, y, 0))
        self.current_x = x
        self.current_y = y
        self.update()

    def rmGazeEffect(self):
        self.waves = []
        self.update()

    def restoreState(self):
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.show()

class GazeThread(QThread):
    gazeSignal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.gaze_estimator = GazeEstimator()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found")
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            x, y = self.gaze_estimator.gaze(frame)
            x = max(50, min(x, 1080 - 50))
            y = max(50, min(y, 1920 - 50))
            self.gazeSignal.emit(x, y)
            cv2.waitKey(1)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class ClickThread(QThread):
    clickSignal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.click_detector = HandClickDetector()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found")
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.click_detector.click(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
                self.clickSignal.emit(True)
            else:
                self.clickSignal.emit(False)
            cv2.waitKey(1)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

def main():
    app = QApplication(sys.argv)
    window = TransparentWindow()

    gaze_thread = GazeThread()
    click_thread = ClickThread()

    is_Clicking_gesture = False

    def update_gaze(x, y):
        window.setGazeCoordinates(x, y)
        pyautogui.moveTo(x, y)

    def update_click(is_clicking):
        nonlocal is_Clicking_gesture
        is_Clicking_gesture = is_clicking
        if is_Clicking_gesture:
            print(f'You clicked at {window.current_x}, {window.current_y}')
            window.rmGazeEffect()
            pyautogui.click(window.current_x, window.current_y)
            window.restoreState()
            is_Clicking_gesture = False

    gaze_thread.gazeSignal.connect(update_gaze)
    click_thread.clickSignal.connect(update_click)

    gaze_thread.start()
    click_thread.start()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
