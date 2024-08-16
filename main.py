import cv2 as cv
import numpy as np
import mediapipe as mp
import argparse
from constants import *
import constants
from gaze import gaze
from display import draw_gaze_point
from AngleBuffer import AngleBuffer
from initial import Affin

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)


cap = cv.VideoCapture(0)
angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)

cv.namedWindow("D")
cv.createTrackbar("distance", "D", 600, 2000, lambda x: x)
cv.setTrackbarPos("distance", "D", 1200)

x_list = [0]
y_list = [0]


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        DISTANCE = cv.getTrackbarPos("distance", "D")
        frame_affin = Affin(frame, (int(sum(x_list) / len(x_list)), int(sum(y_list) / len(y_list))))
        x, y = gaze(frame_affin, DISTANCE, angle_buffer, mp_face_mesh)
        if len(x_list) < NUM_LIST:
            x_list.insert(0, x)
            y_list.insert(0, y)
        else:
            x_list.pop()
            y_list.pop()
            x_list.insert(0, x)
            y_list.insert(0, y)

        gaze_point = [int(sum(x_list) / len(x_list)), int(sum(y_list) / len(y_list))]
        draw_gaze_point(frame, x_list, y_list, gaze_point)
        
        
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            constants.SETUP_STEP = 0
            print("SETUP_STEP is 0")
        if key == ord('s'):
            if(constants.SETUP_STEP == 0) :
                constants.SETUP_STEP = 1
                print("SETUP_STEP is 1")
            elif(constants.SETUP_STEP == 1) :
                constants.SETUP_STEP =2
            elif(constants.SETUP_STEP == 2) :
                constants.SETUP_STEP =3
            elif(constants.SETUP_STEP == 3) :
                constants.SETUP_STEP =4
        if key == ord('r'):
            IS_RECORDING = not IS_RECORDING
            print("Recording started." if IS_RECORDING else "Recording paused.")
        if key == ord('q'):
            print("Exiting program...")
            break
        
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv.destroyAllWindows()