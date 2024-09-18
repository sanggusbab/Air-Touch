import numpy as np
import cv2 as cv
import constants
from constants import *
import math
import mediapipe as mp
import collections
from gaze_f import *

class GazeEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        
        self.x_list = [0]
        self.y_list = [0]
        self.initial = True
        self.calibrated = False
        self.initial_pitch = None
        self.initial_yaw = None
        self.initial_roll = None
        self.initial_diff_eye_x = 0
        self.initial_diff_eye_y = 0
        self.initial_x = 0
        self.initial_y = 0

    def gaze(self, frame):
        if self.initial:
            self.calibrated = 0
            self.initial = False
        
        DISTANCE = 600
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.mp_face_mesh.process(frame)
        
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            (l_bx, l_by), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_bx, r_by), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])        
            (l_ex, l_ey), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_POINTS])
            (r_ex, r_ey), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_POINTS])
            center_eye = np.array([int((l_ex + r_ex) / 2), int((l_ey + r_ey) / 2)], dtype=np.int32)
            diff_eye_x = int((l_ex + r_ex - l_bx - r_bx) / 2)
            diff_eye_y = int((l_ey + r_ey - l_by - r_by) / 2)
            
            if ENABLE_HEAD_POSE:
                pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                # Assuming angle_buffer is defined somewhere
                angle_buffer.add([pitch, yaw, roll])
                pitch, yaw, roll = angle_buffer.get_average()
                
                if self.initial_pitch is None or not self.calibrated:
                    self.initial_pitch, self.initial_yaw, self.initial_roll = pitch, yaw, roll
                    self.initial_x, self.initial_y = img_w / 2 - center_eye[0], img_h / 2 - center_eye[1]
                    self.initial_diff_eye_x, self.initial_diff_eye_y = (diff_eye_x, diff_eye_y) if diff_eye_x is not None and diff_eye_y is not None else (0, 0)
                    self.calibrated = True
                    if PRINT_DATA:
                        print("Head pose recalibrated.")

                if self.calibrated:
                    pitch -= self.initial_pitch
                    yaw -= self.initial_yaw
                    roll -= self.initial_roll
            
            yaw_radian = yaw * np.pi / 180
            pitch_radian = pitch * np.pi / 180
            
            eye_adjustment = [(diff_eye_x - self.initial_diff_eye_x) * EYE_DEGREE_PARAMETER,
                              (-diff_eye_y + self.initial_diff_eye_y) * EYE_DEGREE_PARAMETER * TB_WEIGHT]
            gaze_point = (
                int((self.initial_x + center_eye[0]) / img_w * DISPLAY_W + DISTANCE * math.tan(yaw_radian) + eye_adjustment[0]),
                int((self.initial_y + center_eye[1]) / img_h * DISPLAY_H - DISTANCE * TB_WEIGHT * math.tan(pitch_radian) + eye_adjustment[1]),
            )
            if len(self.x_list) < NUM_LIST:
                self.x_list.insert(0, gaze_point[0])
                self.y_list.insert(0, gaze_point[1])
            else:
                self.x_list.pop()
                self.y_list.pop()
                self.x_list.insert(0, gaze_point[0])
                self.y_list.insert(0, gaze_point[1])
            result_point = [int(sum(self.x_list) / len(self.x_list)), int(sum(self.y_list) / len(self.y_list))]
            
            return result_point[0], result_point[1]
        return 100, 100  # Return None if no face landmarks are detected

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    gaze_estimator = GazeEstimator()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hcd1, hcd2 = gaze_estimator.gaze(frame)
        if hcd1 is not None and hcd2 is not None:
            print(hcd1, hcd2)
