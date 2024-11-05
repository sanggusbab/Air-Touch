import numpy as np
import cv2 as cv
import constants
from constants import *
import math
import mediapipe as mp
import collections
from gaze_f import *


angle_buffer_head = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
angle_buffer_eye = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
angle_buffer_pupils = AngleBuffer(size=MOVING_AVERAGE_WINDOW)

class GazeEstimator:
    def __init__(self):
        
        self.mphands = mp.solutions.hands
        self.my_hands = self.mphands.Hands(static_image_mode=False, max_num_hands=1)
        self.prev_distance = None
        self.click_detected = False
    
        self.initial_tip =[0,0]
        self.initial = True
        self.calibrated = False

    def gaze(self, frame):
        if self.initial:
            self.calibrated = False
            self.initial = False
            
        img_h, img_w = frame.shape[:2]
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.my_hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
            
                center = [handLms.landmark[8].x, handLms.landmark[8].y]
            N = 100

            if not self.calibrated:
                self.initial_tip = center
                self.calibrated = True

            point = [int((-center[0]+self.initial_tip[0])*RATIO+DISPLAY_W/2), int((center[1]-self.initial_tip[1])*RATIO+DISPLAY_H/2)]


            return point
        return 200, 200  # Return None if no face landmarks are detected

    def eye_tracking (self, mesh_points):
        left_eye_points = mesh_points[LEFT_EYE_UNDERPOINTS]
        left_iris_points = mesh_points[LEFT_EYE_IRIS]

        right_eye_points = mesh_points[RIGHT_EYE_UNDERPOINTS]
        right_iris_points = mesh_points[RIGHT_EYE_IRIS]

        # Calculate eye centers
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)

        # Calculate pupil centers
        left_pupil_center = np.mean(left_iris_points, axis=0)
        right_pupil_center = np.mean(right_iris_points, axis=0)
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                       (left_eye_center[1] + right_eye_center[1]) / 2)
        angle =(float(self.pitch * 180.0 / np.pi), float(self.yaw * 180.0 / np.pi))
        M = cv.getRotationMatrix2D(eyes_center, float(self.pitch * 180.0 / np.pi), scale=1.0)



        # Compute differences between eye center and pupil center
        left_diff = left_eye_center - left_pupil_center
        right_diff = right_eye_center - right_pupil_center 
        if not self.calibrated:
            # Store the initial difference as a constant
            self.constant_diff_left = left_diff
            self.constant_diff_right = right_diff
        else:
            # Calculate adjusted differences
            adjusted_left_diff =  np.dot(M,left_diff - self.constant_diff_left)
            adjusted_right_diff = np.dot(M,right_diff - self.constant_diff_right)

            # Output the adjusted differences
            print("Adjusted Left Eye Difference:", adjusted_left_diff)
            print("Adjusted Right Eye Difference:", adjusted_right_diff)
            result = [int((adjusted_right_diff[0] + adjusted_left_diff[0]) * EYE_DEGREE_PARAMETER), int((adjusted_right_diff[1] + adjusted_left_diff[1]) * EYE_DEGREE_PARAMETER*2)]
            return result
        return [0,0]


    def reset(self):
        self.initial = True

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    gaze_estimator = GazeEstimator()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hcd1, hcd2 = gaze_estimator.gaze(frame)
        if hcd1 is not None and hcd2 is not None:
            draw_gaze_point(frame, hcd1, hcd2)
            # print(hcd1, hcd2)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gaze_estimator.reset()