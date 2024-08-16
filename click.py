# click.py
import cv2
import mediapipe as mp
import math
import time

class HandClickDetector:
    def __init__(self, threshold_distance=0.06, debounce_time=0.3):
        self.threshold_distance = threshold_distance
        self.debounce_time = debounce_time
        self.mpHands = mp.solutions.hands
        self.my_hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.prev_distance = None
        self.click_detected = False
        self.click_count = 0
        self.last_click_time = 0

    def dist(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    def click(self, img):
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.my_hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                curdist = self.dist(handLms.landmark[4].x, handLms.landmark[4].y,
                                    handLms.landmark[8].x, handLms.landmark[8].y)
                
                if self.prev_distance is not None:
                    current_time = time.time()
                    if self.prev_distance < self.threshold_distance and curdist > self.threshold_distance: #click
                        if current_time - self.last_click_time > self.debounce_time:  #ignore double click
                            self.click_detected = True
                            self.last_click_time = current_time
                            self.click_count += 1
                        else:
                            self.click_detected = False
                    else:
                        self.click_detected = False
                
                self.prev_distance = curdist

                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return self.click_detected
