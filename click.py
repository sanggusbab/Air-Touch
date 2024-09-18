import mediapipe as mp
import math

class HandClickDetector: 
    def __init__(self, threshold_distance=0.06):
        self.threshold_distance = threshold_distance
    
        self.mpHands = mp.solutions.hands
        self.my_hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=1)
        self.prev_distance = None
        self.click_detected = False

    def dist(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def click(self, imgRGB):
        results = self.my_hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                curdist = self.dist(handLms.landmark[4].x, handLms.landmark[4].y,
                                    handLms.landmark[8].x, handLms.landmark[8].y)
                
                if self.prev_distance is not None:
                    if self.prev_distance < self.threshold_distance and curdist > self.threshold_distance:
                        self.click_detected = True
                    else:
                        self.click_detected = False
                self.prev_distance = curdist
                break  # 첫 번째 손만 처리
        else:
            self.click_detected = False
            self.prev_distance = None
        return self.click_detected
