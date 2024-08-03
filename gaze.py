import numpy as np
import cv2 as cv
from constants import *
from utils import normalize_pitch
import math

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

def gaze(frame, distance, angle_buffer, mp_face_mesh):
    global initial_pitch, initial_yaw, initial_roll, IS_RECORDING, key, calibrated, initial_diff_eye_x, initial_diff_eye_y, initial_y, initial_x
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = mp_face_mesh.process(frame)
    
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
        center_eye = np.array([int((l_ex+r_ex)/2), int((l_ey+ r_ey)/2)], dtype=np.int32)
        
        diff_eye_x = int((l_ex + r_ex - l_bx - r_bx)/2)
        diff_eye_y = int((l_ey + r_ey - l_by - r_by)/2)
        
        if ENABLE_HEAD_POSE:
            pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
            angle_buffer.add([pitch, yaw, roll])
            pitch, yaw, roll = angle_buffer.get_average()

            if initial_pitch is None or (key == ord('c') and calibrated):
                initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                initial_x, initial_y = img_w/2 - center_eye[0],  img_h/2-center_eye[1]
                if diff_eye_x is None or diff_eye_y is None :
                    initial_diff_eye_x, initial_diff_eye_y = 0,0
                else :
                    initial_diff_eye_x, initial_diff_eye_y = diff_eye_x, diff_eye_y
                    
                calibrated = True
                if PRINT_DATA:
                    print("Head pose recalibrated.")

            if calibrated:
                pitch -= initial_pitch
                yaw -= initial_yaw
                roll -= initial_roll
            
        yaw_radian = yaw *  np.pi / 180
        pitch_radian = pitch * np.pi / 180
        
        if SHOW_ON_SCREEN_DATA:
            if IS_RECORDING:
                cv.circle(frame, (30, 30), 10, (0, 0, 255), -1) 
            if ENABLE_HEAD_POSE:
                cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        
        eye_adjustment = [(diff_eye_x - initial_diff_eye_x)*EYE_DEGREE_PARAMETER,(-diff_eye_y + initial_diff_eye_y) * EYE_DEGREE_PARAMETER * TB_WEIGHT ]
        gaze_point = (
            int((initial_x + center_eye[0]) / img_w * DISPLAY_W + distance * math.tan(yaw_radian) + eye_adjustment[0]),
            int((initial_y + center_eye[1]) / img_h * DISPLAY_H - distance * TB_WEIGHT * math.tan(pitch_radian) + eye_adjustment[1]),
        )
        
        return gaze_point[0], gaze_point[1]
