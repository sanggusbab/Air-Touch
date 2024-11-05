import numpy as np
import cv2 as cv
import constants
from constants import *
import math
import mediapipe as mp
import collections
from gaze_f import *


angle_buffer_head = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
angle_buffer_eye = AngleBuffer(size=MOVING_AVERAGE_WINDOW+20)
angle_buffer_pupils = AngleBuffer(size=MOVING_AVERAGE_WINDOW)

class GazeEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        
        self.initial = True
        self.calibrated = False

    def gaze(self, frame):
        if self.initial:
            self.calibrated = False
            self.initial = False
            
        img_h, img_w = frame.shape[:2]
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            
            center = mesh_points[4]
            N = 100
            if ENABLE_HEAD_POSE:
                self.pitch, self.yaw, _ = estimate_head_pose(mesh_points, (img_h, img_w))
                # Assuming angle_buffer is defined somewhere
                angle_buffer_head.add([self.pitch, self.yaw, _])
                self.pitch, self.yaw, _= angle_buffer_head.get_average()
                
                eye_tracking = self.eye_tracking(mesh_points)
                angle_buffer_eye.add([eye_tracking[0],eye_tracking[1]])
                eye_tracking_x, eye_tracking_y= angle_buffer_eye.get_average()
                
                if not self.calibrated:
                    self.initial_pitch, self.initial_yaw = self.pitch, self.yaw
                    self.calibrated = True

                if self.calibrated:
                    self.pitch -= self.initial_pitch
                    self.yaw -= self.initial_yaw
            
            self.yaw_t =  math.tan((self.yaw) * np.pi / 180)
            self.pitch_t = math.tan((self.pitch) * np.pi / 180)
            gaze_point = (
                int((center[0]) / img_w * DISPLAY_W + DISTANCE * self.yaw_t + eye_tracking_x),
                int((center[1]) / img_h * DISPLAY_H - DISTANCE * TB_WEIGHT * self.pitch_t - eye_tracking_y* TB_WEIGHT),
            )

            return gaze_point
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

        # Compute differences between eye center and pupil center
        left_diff = left_eye_center - left_pupil_center
        right_diff = right_eye_center - right_pupil_center 
        if not self.calibrated:
            # Store the initial difference as a constant
            self.constant_diff_left = left_diff
            self.constant_diff_right = right_diff
            
        else :
            result = (-self.constant_diff_left + left_diff) * EYE_DEGREE_PARAMETER + (-self.constant_diff_right + right_diff) * EYE_DEGREE_PARAMETER
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