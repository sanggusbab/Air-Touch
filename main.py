import sys
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
        self.waves = []  # (x, y, radius)
        self.max_radius = 100

        # 물결 효과 업데이트 타이머
        self.wave_timer = QTimer()
        self.wave_timer.timeout.connect(self.update_effect)
        self.wave_timer.start(50)  # 20 FPS

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen_size = pyautogui.size()
        self.setGeometry(0, 0, screen_size.width, screen_size.height)
        self.show()

    def update_effect(self):
        # 물결 효과 반경 증가
        new_waves = []
        for (x, y, radius) in self.waves:
            if radius < self.max_radius:
                new_waves.append((x, y, radius + 5))
        self.waves = new_waves
        self.update()  # repaint 요청

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 배경 투명하게 설정
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # 물결 효과 그리기
        for (x, y, radius) in self.waves:
            alpha = max(0, 127 - int(127 * (radius / self.max_radius)))
            gradient_color = QColor(0, 255, 255, alpha)
            pen = QPen(gradient_color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(x - radius), int(y - radius), int(2 * radius), int(2 * radius))

    def setGazeCoordinates(self, x, y):
        self.waves.append((x, y, 0))
        self.update()

    def rmGazeEffect(self):
        self.waves = []
        self.update()

class VideoThread(QThread):
    frame_updated = pyqtSignal(object)
    gaze_coordinates = pyqtSignal(int, int)
    click_detected = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Webcam not found")
            sys.exit()
        self.hand_click_detector = HandClickDetector()
        self.gaze_estimator = GazeEstimator()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 프레임을 필요한 곳에 전달
            self.frame_updated.emit(frame)

            # 시선 추정
            x, y = self.gaze_estimator.gaze(frame)
            screen_width, screen_height = pyautogui.size()
            x = max(50, min(x, screen_width - 50))
            y = max(50, min(y, screen_height - 50))
            self.gaze_coordinates.emit(x, y)

            # 마우스 이동
            pyautogui.moveTo(x, y)

            # 손 클릭 감지
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.hand_click_detector.click(img_rgb):
                self.click_detected.emit()

            # CPU 사용량을 낮추기 위해 잠시 대기
            self.msleep(10)  # 100 FPS 정도로 루프 실행

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()
        self.wait()

def main():
    app = QApplication(sys.argv)
    window = TransparentWindow()

    video_thread = VideoThread()

    # 시선 좌표 업데이트 시그널 연결
    video_thread.gaze_coordinates.connect(window.setGazeCoordinates)

    # 클릭 감지 시그널 연결
    def handle_click():
        x, y = pyautogui.position()
        window.rmGazeEffect()
        pyautogui.click(x, y)
        window.restoreState()
    video_thread.click_detected.connect(handle_click)

    video_thread.start()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
