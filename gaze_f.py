import numpy as np
import cv2 as cv
import constants
from constants import *
import math
import mediapipe as mp
import collections

def draw_gaze_point(frame, x_list, y_list, gaze_point):
    
    display = np.zeros((DISPLAY_H, DISPLAY_W), np.uint8)
    cv.circle(display, gaze_point, 10, (255, 0, 0), 5, cv.LINE_AA)
    cv.circle(display, (x_list[0], y_list[0]), 5, (255, 0, 0), 5, cv.LINE_AA)
    cv.putText(display, f"gaze_point: {gaze_point}", (100, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (182, 242, 32), 2, cv.LINE_AA)

    for i in range(NUM_GRID):
        tmp_w = int(DISPLAY_W * i / NUM_GRID)
        tmp_h = int(DISPLAY_H * i / NUM_GRID)
        cv.line(display, (tmp_w, 0), (tmp_w, DISPLAY_H), (255, 255, 255))
        cv.line(display, (0, tmp_h), (DISPLAY_W, tmp_h), (255, 255, 255))
    cv.circle(display, (int(DISPLAY_W / 2), int(DISPLAY_H / 2)), 5, (255, 0, 0), 1, cv.LINE_AA)

    gaze_grid = [int(gaze_point[0] * NUM_GRID / DISPLAY_W), int(gaze_point[1] * NUM_GRID / DISPLAY_H)]
    cv.rectangle(display, (int(DISPLAY_W * gaze_grid[0] / NUM_GRID), int(DISPLAY_H * gaze_grid[1] / NUM_GRID)),
                 (int(DISPLAY_W * (gaze_grid[0] + 1) / NUM_GRID), int(DISPLAY_H * (gaze_grid[1] + 1) / NUM_GRID)), (255, 0, 0), 5)
    cv.imshow("display", display)
    cv.imshow("EyeTracking", frame)


def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1

def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    distance = numerator / denominator
    return distance

def normalize_pitch(pitch):
    if pitch > 180:
        pitch -= 360
    pitch = -pitch
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
    pitch = -pitch
    return pitch



class AngleBuffer:
    def __init__(self, size=40):
        self.size = size
        self.buffer = collections.deque(maxlen=size)

    def add(self, angles):
        self.buffer.append(angles)

    def get_average(self):
        return np.mean(self.buffer, axis=0)
    
angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
    

def estimate_head_pose(landmarks, image_size):
    
    scale_factor = USER_FACE_WIDTH / 150.0
    model_points = np.array([
        (0.0, 0.0, 0.0),             
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      
    ])
    
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    dist_coeffs = np.zeros((4,1))

    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],            
        landmarks[CHIN_INDEX],                
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  
        landmarks[LEFT_MOUTH_CORNER_INDEX],      
        landmarks[RIGHT_MOUTH_CORNER_INDEX]      
    ], dtype="double")

    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]

    pitch = normalize_pitch(pitch)

    return pitch, yaw, roll