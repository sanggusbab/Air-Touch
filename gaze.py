import numpy as np
import cv2 as cv
import constants
from constants import *
import math
import mediapipe as mp
import collections
from gaze_f import *


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

x_list = [0]
y_list = [0]
initial =True

def gaze(frame):
    
    global initial_pitch, initial_yaw, initial_roll, calibrated, initial_diff_eye_x, initial_diff_eye_y, initial_y, initial_x, Point_affin, key, SETUP_STEP, is_Clicking_gesture
    
    # frame_affin = Affin(frame, (int(sum(x_list) / len(x_list)), int(sum(y_list) / len(y_list))))
    if(initial):
        if(is_Clicking_gesture) :
            calibrated =0
            initial =False
    
    DISTANCE = 600
    # frame_affin = Affin(frame, (int(sum(x_list) / len(x_list)), int(sum(y_list) / len(y_list))))
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
        print(center_eye)
        diff_eye_x = int((l_ex + r_ex - l_bx - r_bx)/2)
        diff_eye_y = int((l_ey + r_ey - l_by - r_by)/2)
        
        if ENABLE_HEAD_POSE:
            pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
            angle_buffer.add([pitch, yaw, roll])
            pitch, yaw, roll = angle_buffer.get_average()
            # if initial_pitch is None or1 (constants.SETUP_STEP == 0 and calibrated):
            if initial_pitch is None or (not calibrated):

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
            if ENABLE_HEAD_POSE:
                cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        
        eye_adjustment = [(diff_eye_x - initial_diff_eye_x)*EYE_DEGREE_PARAMETER,(-diff_eye_y + initial_diff_eye_y) * EYE_DEGREE_PARAMETER * TB_WEIGHT ]
        gaze_point = (
            int((initial_x + center_eye[0]) / img_w * DISPLAY_W + DISTANCE * math.tan(yaw_radian) + eye_adjustment[0]),
            int((initial_y + center_eye[1]) / img_h * DISPLAY_H - DISTANCE * TB_WEIGHT * math.tan(pitch_radian) + eye_adjustment[1]),
        )
        if len(x_list) < NUM_LIST:
            x_list.insert(0, gaze_point[0])
            y_list.insert(0, gaze_point[1])
        else:
            x_list.pop()
            y_list.pop()
            x_list.insert(0, gaze_point[0])
            y_list.insert(0, gaze_point[1])
        result_point = [int(sum(x_list) / len(x_list)), int(sum(y_list) / len(y_list))]
        
        return result_point[0],result_point[1]

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    
    while True :
        
        ret, frame = cap.read() 
        hcd1,hcd2 = gaze(frame)
        print(hcd1, hcd2)
    

