import numpy as np
import cv2 as cv
import constants
from constants import *
import math
import mediapipe as mp
import collections

def draw_gaze_point(frame, X, Y):
    display = np.zeros((DISPLAY_H, DISPLAY_W), np.uint8)
    frame2show = frame
    cv.circle(display, (X, Y), 5, (255, 0, 0), 5, cv.LINE_AA)
    cv.circle(frame2show, (X, Y), 5, (255, 0, 0), 5, cv.LINE_AA)

    cv.imshow("display", display)
    cv.imshow("EyeTracking", frame2show)


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

def estimate_gaze_direction(landmarks, image_size):
    """
    동공과 눈 주변 랜드마크를 사용하여 시선 방향(피치, 요, 롤)을 추정합니다.

    :param landmarks: 얼굴 랜드마크 배열
    :param image_size: 이미지 크기 (폭, 높이)
    :return: 피치, 요, 롤 각도
    """

    # 사용자의 눈 크기에 따른 스케일 조정
    scale_factor = USER_EYE_WIDTH / 30.0  # USER_EYE_WIDTH는 사용자 눈의 폭(mm)

    # 3D 모델 포인트 정의 (눈 주변)
    model_points = np.array([
        (0.0, 0.0, 0.0),                              # 동공 중심
        (-15.0 * scale_factor, 0.0, 0.0),             # 왼쪽 눈꼬리
        (15.0 * scale_factor, 0.0, 0.0),              # 오른쪽 눈꼬리
        (0.0, -10.0 * scale_factor, 0.0),             # 눈 위쪽 중심
        (0.0, 10.0 * scale_factor, 0.0),              # 눈 아래쪽 중심
        (0.0, 0.0, 0.0),                              # 동공 중심 (반복)
    ])

    # 카메라 내부 파라미터
    focal_length = image_size[0]
    center = (image_size[0] / 2, image_size[1] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # 왜곡 없음 가정

    # 왼쪽 눈 이미지 포인트
    left_eye_image_points = np.array([
        landmarks[LEFT_PUPIL_INDEX],                  # 동공 중심
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],        # 왼쪽 눈꼬리
        landmarks[LEFT_EYE_RIGHT_CORNER_INDEX],       # 오른쪽 눈꼬리
        landmarks[LEFT_EYE_TOP_CENTER_INDEX],         # 눈 위쪽 중심
        landmarks[LEFT_EYE_BOTTOM_CENTER_INDEX],      # 눈 아래쪽 중심
        landmarks[LEFT_PUPIL_INDEX],                  # 동공 중심 (반복)
    ], dtype="double")

    # 오른쪽 눈 이미지 포인트
    right_eye_image_points = np.array([
        landmarks[RIGHT_PUPIL_INDEX],                 # 동공 중심
        landmarks[RIGHT_EYE_LEFT_CORNER_INDEX],       # 왼쪽 눈꼬리
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],      # 오른쪽 눈꼬리
        landmarks[RIGHT_EYE_TOP_CENTER_INDEX],        # 눈 위쪽 중심
        landmarks[RIGHT_EYE_BOTTOM_CENTER_INDEX],     # 눈 아래쪽 중심
        landmarks[RIGHT_PUPIL_INDEX],                 # 동공 중심 (반복)
    ], dtype="double")

    # 왼쪽 눈 포즈 추정
    success_left, rotation_vector_left, translation_vector_left = cv.solvePnP(
        model_points, left_eye_image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
    )

    # 오른쪽 눈 포즈 추정
    success_right, rotation_vector_right, translation_vector_right = cv.solvePnP(
        model_points, right_eye_image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
    )

    # 회전 벡터를 회전 행렬로 변환
    rotation_matrix_left, _ = cv.Rodrigues(rotation_vector_left)
    rotation_matrix_right, _ = cv.Rodrigues(rotation_vector_right)

    # 오일러 각도로 변환
    euler_angles_left = cv.decomposeProjectionMatrix(
        np.hstack((rotation_matrix_left, translation_vector_left))
    )[-1]
    pitch_left, yaw_left, roll_left = euler_angles_left.flatten()[:3]

    euler_angles_right = cv.decomposeProjectionMatrix(
        np.hstack((rotation_matrix_right, translation_vector_right))
    )[-1]
    pitch_right, yaw_right, roll_right = euler_angles_right.flatten()[:3]

    # 피치 각도 정규화
    pitch_left = normalize_pitch(pitch_left)
    pitch_right = normalize_pitch(pitch_right)

    # 좌우 눈의 각도 평균 계산
    pitch = (pitch_left + pitch_right) / 2
    yaw = (yaw_left + yaw_right) / 2
    roll = (roll_left + roll_right) / 2

    return pitch, yaw, roll


def estimate_gaze_and_head_pose(landmarks, image_size):
    """
    동공과 눈 주변 랜드마크를 사용하여 시선 방향(피치, 요, 롤)을 추정하고,
    얼굴 랜드마크를 사용하여 머리 자세(피치, 요, 롤)을 추정합니다.

    :param landmarks: 얼굴 랜드마크 배열
    :param image_size: 이미지 크기 (폭, 높이)
    :return: 시선 피치, 시선 요, 시선 롤, 머리 피치, 머리 요, 머리 롤 각도
    """

    # 사용자 정의 상수 (사용자의 눈과 얼굴 폭(mm))
    USER_EYE_WIDTH = 30.0  # 예시 값, 실제 값으로 대체해야 함
    USER_FACE_WIDTH = 150.0  # 예시 값, 실제 값으로 대체해야 함

    # 랜드마크 인덱스 정의 (실제 인덱스로 대체해야 함)
    LEFT_PUPIL_INDEX = 0
    LEFT_EYE_LEFT_CORNER_INDEX = 1
    LEFT_EYE_RIGHT_CORNER_INDEX = 2
    LEFT_EYE_TOP_CENTER_INDEX = 3
    LEFT_EYE_BOTTOM_CENTER_INDEX = 4
    RIGHT_PUPIL_INDEX = 5
    RIGHT_EYE_LEFT_CORNER_INDEX = 6
    RIGHT_EYE_RIGHT_CORNER_INDEX = 7
    RIGHT_EYE_TOP_CENTER_INDEX = 8
    RIGHT_EYE_BOTTOM_CENTER_INDEX = 9
    NOSE_TIP_INDEX = 10
    CHIN_INDEX = 11
    LEFT_MOUTH_CORNER_INDEX = 12
    RIGHT_MOUTH_CORNER_INDEX = 13

    # 시선 방향 추정
    scale_factor_eye = USER_EYE_WIDTH / 30.0

    model_points_eye = np.array([
        (0.0, 0.0, 0.0),                              
        (-15.0 * scale_factor_eye, 0.0, 0.0),         
        (15.0 * scale_factor_eye, 0.0, 0.0),          
        (0.0, -10.0 * scale_factor_eye, 0.0),         
        (0.0, 10.0 * scale_factor_eye, 0.0),          
        (0.0, 0.0, 0.0),                              
    ])

    focal_length = image_size[0]
    center = (image_size[0] / 2, image_size[1] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    left_eye_image_points = np.array([
        landmarks[LEFT_PUPIL_INDEX],                  
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],        
        landmarks[LEFT_EYE_RIGHT_CORNER_INDEX],       
        landmarks[LEFT_EYE_TOP_CENTER_INDEX],         
        landmarks[LEFT_EYE_BOTTOM_CENTER_INDEX],      
        landmarks[LEFT_PUPIL_INDEX],                  
    ], dtype="double")

    right_eye_image_points = np.array([
        landmarks[RIGHT_PUPIL_INDEX],                 
        landmarks[RIGHT_EYE_LEFT_CORNER_INDEX],       
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],      
        landmarks[RIGHT_EYE_TOP_CENTER_INDEX],        
        landmarks[RIGHT_EYE_BOTTOM_CENTER_INDEX],     
        landmarks[RIGHT_PUPIL_INDEX],                 
    ], dtype="double")

    success_left, rotation_vector_left, translation_vector_left = cv.solvePnP(
        model_points_eye, left_eye_image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
    )

    success_right, rotation_vector_right, translation_vector_right = cv.solvePnP(
        model_points_eye, right_eye_image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
    )

    rotation_matrix_left, _ = cv.Rodrigues(rotation_vector_left)
    rotation_matrix_right, _ = cv.Rodrigues(rotation_vector_right)

    euler_angles_left = cv.decomposeProjectionMatrix(
        np.hstack((rotation_matrix_left, translation_vector_left))
    )[-1]
    gaze_pitch_left, gaze_yaw_left, gaze_roll_left = euler_angles_left.flatten()[:3]

    euler_angles_right = cv.decomposeProjectionMatrix(
        np.hstack((rotation_matrix_right, translation_vector_right))
    )[-1]
    gaze_pitch_right, gaze_yaw_right, gaze_roll_right = euler_angles_right.flatten()[:3]

    # 피치 각도 정규화 함수 (필요에 따라 구현)
    def normalize_pitch(pitch):
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return pitch

    gaze_pitch_left = normalize_pitch(gaze_pitch_left)
    gaze_pitch_right = normalize_pitch(gaze_pitch_right)

    gaze_pitch = (gaze_pitch_left + gaze_pitch_right) / 2
    gaze_yaw = (gaze_yaw_left + gaze_yaw_right) / 2
    gaze_roll = (gaze_roll_left + gaze_roll_right) / 2

    # 머리 자세 추정
    scale_factor_face = USER_FACE_WIDTH / 150.0

    model_points_face = np.array([
        (0.0, 0.0, 0.0),                                                 
        (0.0, -330.0 * scale_factor_face, -65.0 * scale_factor_face),    
        (-225.0 * scale_factor_face, 170.0 * scale_factor_face, -135.0 * scale_factor_face),  
        (225.0 * scale_factor_face, 170.0 * scale_factor_face, -135.0 * scale_factor_face),   
        (-150.0 * scale_factor_face, -150.0 * scale_factor_face, -125.0 * scale_factor_face), 
        (150.0 * scale_factor_face, -150.0 * scale_factor_face, -125.0 * scale_factor_face)   
    ])

    image_points_face = np.array([
        landmarks[NOSE_TIP_INDEX],              
        landmarks[CHIN_INDEX],                  
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],
        landmarks[LEFT_MOUTH_CORNER_INDEX],     
        landmarks[RIGHT_MOUTH_CORNER_INDEX],    
    ], dtype="double")

    success, rotation_vector, translation_vector = cv.solvePnP(
        model_points_face, image_points_face, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    projection_matrix = np.hstack((rotation_matrix, translation_vector))

    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    head_pitch, head_yaw, head_roll = euler_angles.flatten()[:3]

    head_pitch = normalize_pitch(head_pitch)

    return gaze_pitch, gaze_yaw, gaze_roll, head_pitch, head_yaw, head_roll