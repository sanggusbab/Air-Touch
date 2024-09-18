# main.py
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys
import cv2
import pyautogui
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush
from click import HandClickDetector
from gaze import GazeEstimator

class TransparentWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
     
        self.setGeometry(0, 0, 1080, 1920)
        self.show()

        # 물결 효과 관련 변수
        self.waves = []  # (x, y, radius) 형태의 튜플을 저장
        self.max_radius = 100
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_effect)
        self.timer.start(50)

    def update_effect(self):
        # 각 물결의 반경을 증가시킴
        new_waves = []
        for (x, y, radius) in self.waves:
            if radius < self.max_radius:
                new_waves.append((x, y, radius + 5))
        self.waves = new_waves
        self.update()  # 화면을 갱신하여 물결 효과를 다시 그림

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 배경을 투명하게 설정
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # 물결 효과 그리기
        for (x, y, radius) in self.waves:
            alpha = max(0, 127 - int(127 * (radius / self.max_radius)))  # 반투명도 조정
            gradient_color = QColor(0, 255, 255, alpha)
            pen = QPen(gradient_color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(x - radius, y - radius, 2 * radius, 2 * radius)

    def setGazeCoordinates(self, x, y):
        self.waves.append((x, y, 0))  # 새로운 좌표에서 물결 효과 시작
        self.update()  # 새로운 좌표에서 즉시 물결 효과가 나타나도록 업데이트

    def rmGazeEffect(self):
        self.waves = []  # 모든 물결을 제거하여 효과를 없앰
        self.update()
    
    def restoreState(self):
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.show()

# def gaze(_frame=None):
#     import random
#     random_num = random.randint(0, 10) % 10
#     if random_num > 5:
#         x = random.randint(100, 500)
#         y = random.randint(100, 1000)
#     else:
#         x = 200
#         y = 200
#     return x, y

def main():
    app = QApplication(sys.argv)
    window = TransparentWindow()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found")
        return
    hand_click_detector = HandClickDetector()
    Gaze = GazeEstimator()
    is_Clicking_gesture = False
    x = 0
    y = 0

    cycleCounter = 0

    gazePeriod = 1
    gesturePeriod = 1
    windowPeriod = 5
    
    while True:
        cycleCounter += 1
        ret, frame = cap.read()

        ##########################################################################################
        ##########################################################################################
        if 0 == cycleCounter % gazePeriod: # 10 * N/sec executing
            x, y = Gaze.gaze(frame) # TODO: develop gaze module
            x = max(50, min(x, 1080 - 50))
            y = max(50, min(y, 1920 - 50))

        ##########################################################################################
        ##########################################################################################

        print(x, y)
        pyautogui.moveTo(x, y)

        ##########################################################################################
        ##########################################################################################
        if 0 == cycleCounter % gesturePeriod: # 15 * N/sec executing
            if hand_click_detector.click(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
                is_Clicking_gesture = True  # TODO: develop gaze module
        ##########################################################################################
        ##########################################################################################

        if 0 == cycleCounter % windowPeriod: # 6 * N/sec executing
            if is_Clicking_gesture:
                print(f'you clicked {x}, {y}')
                window.rmGazeEffect()
                pyautogui.click(x, y)
                window.restoreState()
                is_Clicking_gesture = False
            else:
                window.setGazeCoordinates(x, y)
        app.processEvents()

if __name__ == '__main__':
    main()