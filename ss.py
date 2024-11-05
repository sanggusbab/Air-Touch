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
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        
        self.initial = True
        self.calibrated = False
        # Set the initial gaze point to the center of the screen
        self.initial_gaze_point = (DISPLAY_W // 2, DISPLAY_H // 2)
        self.constant_diff_left = None
        self.constant_diff_right = None

    def gaze(self, frame):
        if self.initial:
            # During initial setup, set the gaze to the center of the screen and proceed with calibration
            self.initial = False
            self.calibrated = False
            return self.initial_gaze_point
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
                
                eye_tracking = self.eye_tracking(mesh_points, frame)
                angle_buffer_eye.add([eye_tracking[0], eye_tracking[1]])
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
                int((center[1]) / img_h * DISPLAY_H - DISTANCE * TB_WEIGHT * self.pitch_t - eye_tracking_y * TB_WEIGHT),
            )

            return gaze_point
        return 200, 200  # Return None if no face landmarks are detected

    def eye_tracking(self, mesh_points, frame):
        # Get eye regions (left and right)
        left_eye_points = mesh_points[LEFT_EYE_UNDERPOINTS]
        right_eye_points = mesh_points[RIGHT_EYE_UNDERPOINTS]

        # Extract eye region from the frame
        left_eye_rect = cv.boundingRect(left_eye_points)
        right_eye_rect = cv.boundingRect(right_eye_points)

        left_eye_crop = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3],
                              left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
        right_eye_crop = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3],
                               right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]

        # Process each eye to detect white region
        left_gaze_point = self.process_eye(left_eye_crop)
        right_gaze_point = self.process_eye(right_eye_crop)

        # Use the eye with the larger size for gaze estimation
        left_eye_size = np.linalg.norm(left_eye_points[3] - left_eye_points[0])
        right_eye_size = np.linalg.norm(right_eye_points[3] - right_eye_points[0])

        if left_eye_size >= right_eye_size:
            chosen_gaze_point = left_gaze_point
            print("Using Left Eye for Gaze Estimation")
        else:
            chosen_gaze_point = right_gaze_point
            print("Using Right Eye for Gaze Estimation")

        if chosen_gaze_point is not None:
            return chosen_gaze_point *(-10)

        return [0, 0]

    def process_eye(self, eye_crop):
        if eye_crop.size == 0:
            return None
        
        # Convert to grayscale
        gray_eye = cv.cvtColor(eye_crop, cv.COLOR_BGR2GRAY)

        # Apply threshold to extract white region
        _, thresh_eye = cv.threshold(gray_eye, 70, 255, cv.THRESH_BINARY)

        # Find all white points
        white_points = np.argwhere(thresh_eye == 255)

        if len(white_points) > 0:
            # Calculate the mean of white points
            mean_point = np.mean(white_points, axis=0).astype(int)
            return mean_point
        else:
            return None

    def reset(self):
        self.initial = True
        self.calibrated = False
        self.constant_diff_left = None
        self.constant_diff_right = None

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    gaze_estimator = GazeEstimator()
    
    # Create a black background to draw gaze estimation points
    background = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gaze_x, gaze_y = gaze_estimator.gaze(frame)
        if gaze_x is not None and gaze_y is not None:
            # Draw the gaze point on the black background
            background[:] = 0  # Clear the background
            cv.circle(background, (gaze_x, gaze_y), 5, (0, 255, 0), -1)

        # Show the gaze estimation on the black background
        cv.imshow('Gaze Estimation', background)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gaze_estimator.reset()

    cap.release()
    cv.destroyAllWindows()
