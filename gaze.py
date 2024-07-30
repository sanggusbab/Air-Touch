import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os
from AngleBuffer import AngleBuffer
import tkinter


#-----------------------------------------------------------------------------------------------------------------------------------
# The vector sum was calculated in the same way for both face and eye gaze. 

#Function name: gaze
# input: frame, distance
#output: gaze_point_x, gaze_point_y

## How it works 

# Gaze at the center dot and press 'c' to set that point as the initial value. 
# Based on the initial value, it shows the coordinates moved by the line of sight.
# The large dot is the average position of the previous 10 coordinates (NUM_LIST), and the small dot is the most recent one. 
# There were a lot of them, so I decided on 10, but I think I need to reduce them depending on the calculation speed. 

#Face: Measure the orthographic angle using the 2D and 3D coordinates of the face -> Find the coordinates and angle of the face. 

#Pupil: Compare the center of the pupil and the center of the black ruler to find the coordinates of which direction you are looking 
# and multiply by an appropriate constant. (EYE_DEGREE_PARAMETER)  The eyes move too little, making it difficult to do so in the same way as the face. 

#current situation: 
#Currently, left and right are still recognized well, but it seems that top and bottom (especially bottom) are not recognized well. 
#Currently, the top and bottom are weighted, but it would be nice if there was a better way.
#-----------------------------------------------------------------------------------------------------------------------------------
# Parameters Documentation

## Camera Parameters (not currently used in calculations)
# NOSE_TO_CAMERA_DISTANCE: The distance from the tip of the nose to the camera lens in millimeters.
# Intended for future use where accurate physical distance measurements may be necessary.
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]
DISTANCE = 600

# monitor size
root = tkinter.Tk()
DISPLAY_H = root.winfo_screenheight() -50
DISPLAY_W = root.winfo_screenwidth()

#num of print grid
NUM_GRID = 5 
NUM_LIST = 10
# eye gaze parameter
EYE_DEGREE_PARAMETER = DISTANCE / 10

TB_WEIGHT = 1.2


## User-Specific Measurements
# USER_FACE_WIDTH: The horizontal distance between the outer edges of the user's cheekbones in millimeters. 
# This measurement is used to scale the 3D model points for head pose estimation.
# Measure your face width and adjust the value accordingly.
USER_FACE_WIDTH = 140  # [mm]

## Configuration Parameters
# PRINT_DATA: Enable or disable the printing of data to the console for debugging.
PRINT_DATA = True

# DEFAULT_WEBCAM: Default camera source index. '0' usually refers to the built-in webcam.
DEFAULT_WEBCAM = 0

# SHOW_ALL_FEATURES: If True, display all facial landmarks on the video feed.
SHOW_ALL_FEATURES = True

# LOG_DATA: Enable or disable logging of data to a CSV file.
LOG_DATA = True

# LOG_ALL_FEATURES: If True, log all facial landmarks to the CSV file.
LOG_ALL_FEATURES = False

# ENABLE_HEAD_POSE: Enable the head position and orientation estimator.
ENABLE_HEAD_POSE = True

## Logging Configuration
# LOG_FOLDER: Directory where log files will be stored.
LOG_FOLDER = "logs"

## Server Configuration
# SERVER_IP: IP address of the server for sending data via UDP (default is localhost).
SERVER_IP = "127.0.0.1"

# SERVER_PORT: Port number for the server to listen on.
SERVER_PORT = 7070

## Blink Detection Parameters
# SHOW_ON_SCREEN_DATA: If True, display blink count and head pose angles on the video feed.
SHOW_ON_SCREEN_DATA = True

# TOTAL_BLINKS: Counter for the total number of blinks detected.
TOTAL_BLINKS = 0

# EYES_BLINK_FRAME_COUNTER: Counter for consecutive frames with detected potential blinks.
EYES_BLINK_FRAME_COUNTER = 0

# BLINK_THRESHOLD: Eye aspect ratio threshold below which a blink is registered.
BLINK_THRESHOLD = 0.51

# EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold required to confirm a blink.
EYE_AR_CONSEC_FRAMES = 2

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

## MediaPipe Model Confidence Parameters
# These thresholds determine how confidently the model must detect or track to consider the results valid.
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

## Angle Normalization Parameters
# MOVING_AVERAGE_WINDOW: The number of frames over which to calculate the moving average for smoothing angles.
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
# initial_pitch, initial_yaw, initial_roll: Store the initial head pose angles for calibration purposes.
# calibrated: A flag indicating whether the initial calibration has been performed.
initial_pitch, initial_yaw, initial_roll = None, None, None
initial_diff_eye_x,initial_diff_eye_y = None,None
initial_x, initial_y = None, None
calibrated = False

# User-configurable parameters
PRINT_DATA = True  # Enable/disable data printing
DEFAULT_WEBCAM = 0  # Default webcam number
SHOW_ALL_FEATURES = True  # Show all facial landmarks if True
LOG_DATA = True  # Enable logging to CSV
LOG_ALL_FEATURES = False  # Log all facial landmarks if True
LOG_FOLDER = "logs"  # Folder to store log files

# Server configuration
SERVER_IP = "127.0.0.1"  # Set the server IP address (localhost)
SERVER_PORT = 7070  # Set the server port

# eyes blinking variables
SHOW_BLINK_COUNT_ON_SCREEN = True  # Toggle to show the blink count on the video feed
TOTAL_BLINKS = 0  # Tracks the total number of blinks detected
EYES_BLINK_FRAME_COUNTER = (
    0  # Counts the number of consecutive frames with a potential blink
)
BLINK_THRESHOLD = 0.51  # Threshold for the eye aspect ratio to trigger a blink
EYE_AR_CONSEC_FRAMES = (
    2  # Number of consecutive frames below the threshold to confirm a blink
)
# SERVER_ADDRESS: Tuple containing the SERVER_IP and SERVER_PORT for UDP communication.
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)


#If set to false it will wait for your command (hittig 'r') to start logging.
IS_RECORDING = False  # Controls whether data is being logged

# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
)
args = parser.parse_args()

# Iris and eye corners landmarks indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # Left eye Left Corner
L_H_RIGHT = [133]  # Left eye Right Corner
R_H_LEFT = [362]  # Right eye Left Corner
R_H_RIGHT = [263]  # Right eye Right Corner

# Blinking Detection landmark's indices.
# P0, P3, P4, P5, P8, P11, P12, P13
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]

# Server address for UDP socket communication
SERVER_ADDRESS = (SERVER_IP, 7070)

# Function to calculate vector position
def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1
def euclidean_distance_3D(points):
    """Calculates the Euclidean distance between two points in 3D space.

    Args:
        points: A list of 3D points.

    Returns:
        The Euclidean distance between the two points.

        # Comment: This function calculates the Euclidean distance between two points in 3D space.
    """

    # Get the three points.
    P0, P3, P4, P5, P8, P11, P12, P13 = points

    # Calculate the numerator.
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )

    # Calculate the denominator.
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3

    # Calculate the distance.
    distance = numerator / denominator

    return distance
def estimate_head_pose(landmarks, image_size):
    # Scale factor based on user's face width (assumes model face width is 150mm)
    scale_factor = USER_FACE_WIDTH / 150.0
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
    ])
    

    # Camera internals
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1))

    # 2D image points from landmarks, using defined indices
    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],            # Nose tip
        landmarks[CHIN_INDEX],                # Chin
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
        landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
        landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
    ], dtype="double")


        # Solve for pose
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector to form a 3x4 projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

    # Decompose the projection matrix to extract Euler angles
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]


     # Normalize the pitch angle
    pitch = normalize_pitch(pitch)

    return pitch, yaw, roll
def normalize_pitch(pitch):
    """
    Normalize the pitch angle to be within the range of [-90, 90].

    Args:
        pitch (float): The raw pitch angle in degrees.

    Returns:
        float: The normalized pitch angle.
    """
    # Map the pitch angle to the range [-180, 180]
    if pitch > 180:
        pitch -= 360

    # Invert the pitch angle for intuitive up/down movement
    pitch = -pitch

    # Ensure that the pitch is within the range of [-90, 90]
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
        
    pitch = -pitch

    return pitch

def gaze(frame, distance):
    global initial_pitch, initial_yaw, initial_roll, IS_RECORDING, key,calibrated, initial_diff_eye_x, initial_diff_eye_y,initial_y,initial_x
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = mp_face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        mesh_points = np.array(
            [
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ]
        )

        # eye
        (l_bx, l_by), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
        (r_bx, r_by), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])        
        (l_ex, l_ey), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_POINTS])
        (r_ex, r_ey), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_POINTS])
        center_eye = np.array([int((l_ex+r_ex)/2), int((l_ey+ r_ey)/2)], dtype=np.int32)
        
        diff_eye_x = int((l_ex + r_ex - l_bx - r_bx)/2)
        diff_eye_y = int((l_ey + r_ey - l_by - r_by)/2)
        
        # head pose -> pitch, yaw
        if ENABLE_HEAD_POSE:
            pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
            angle_buffer.add([pitch, yaw, roll])
            pitch, yaw, roll = angle_buffer.get_average()

            # Set initial angles on first successful estimation or recalibrate
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

            # Adjust angles based on initial calibration
            if calibrated:
                pitch -= initial_pitch
                yaw -= initial_yaw
                roll -= initial_roll
            
            
            # if PRINT_DATA:
            #     print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
        
        yaw_radian = yaw *  np.pi / 180
        pitch_radian = pitch * np.pi / 180
        
        # print in frame
        if SHOW_ON_SCREEN_DATA:
            if IS_RECORDING:
                cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle at the top-left corner
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

# Initializing MediaPipe face mesh and camera
if PRINT_DATA:
    print("Initializing the face mesh and camera...")
    if PRINT_DATA:
        head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
        print(f"Head pose estimation is {head_pose_status}.")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)
# Main loop for video capture and processing
try:
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing
    x_list = []
    y_list = []
    
    #define disteance 
    cv.namedWindow("D")
    cv.createTrackbar("distance", "D", 600, 2000, lambda x: x)
    cv.setTrackbarPos("distance", "D", 1200)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        DISTANCE = cv.getTrackbarPos("distance", "D")
        
        #gaze
        x, y = gaze(frame, DISTANCE)
        
        #gaze point list
        if(len(x_list) < NUM_LIST):
            x_list.insert(0,x)
            y_list.insert(0,y)
        else : 
            x_list.pop()
            y_list.pop()
            x_list.insert(0,x)
            y_list.insert(0,y)
        
        gaze_point = [int((sum(x_list)/len(x_list))), int((sum(y_list)/len(y_list)))]
        display = np.zeros((DISPLAY_H,DISPLAY_W),np.uint8)
        cv.circle(display, gaze_point, 10, (255, 0, 0), 5, cv.LINE_AA)
        cv.circle(display, (x,y), 5, (255, 0, 0), 5, cv.LINE_AA)
        cv.putText(display, f"gaze_point: {gaze_point}", (100, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (182, 242, 32), 2, cv.LINE_AA)

        # grid
        for i in range (NUM_GRID):
            tmp_w = int(DISPLAY_W*i/NUM_GRID)
            tmp_h = int(DISPLAY_H*i/NUM_GRID)
            cv.line(display, (tmp_w,0),(tmp_w,DISPLAY_H),(255,255,255))
            cv.line(display, (0,tmp_h),(DISPLAY_W,tmp_h),(255,255,255))
        cv.circle(display,(int(DISPLAY_W/2), int(DISPLAY_H/2)),5,(255,0,0),1, cv.LINE_AA)        
        
        gaze_grid = [int(gaze_point[0] * NUM_GRID /DISPLAY_W), int(gaze_point[1] * NUM_GRID /DISPLAY_H)]
        # print(gaze_grid)
        
        cv.rectangle(display, (int(DISPLAY_W *gaze_grid[0]/NUM_GRID), int(DISPLAY_H * gaze_grid[1]/NUM_GRID) ), (int(DISPLAY_W *(gaze_grid[0]+1)/NUM_GRID),int(DISPLAY_H * (gaze_grid[1]+1)/NUM_GRID)), (255,0,0),5)
        cv.imshow("display",display)
        
        
        # Displaying the processed frame
        cv.imshow("Eye Tracking", frame)
        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        
        # Inside the main loop, handle the 'r' key press
        if key == ord('r'):
            
            IS_RECORDING = not IS_RECORDING
            if IS_RECORDING:
                print("Recording started.")
            else:
                print("Recording paused.")
        
        if key == ord('q'):
            if PRINT_DATA:
                print("Exiting program...")
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Releasing camera and closing windows
    cap.release()
    cv.destroyAllWindows()   
