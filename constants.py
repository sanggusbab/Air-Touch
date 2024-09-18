import tkinter

# Camera Parameters
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]
DISTANCE = 1000

# Monitor size
root = tkinter.Tk()
DISPLAY_H = root.winfo_screenheight() - 50
DISPLAY_W = root.winfo_screenwidth()
SETUP_STEP = 0
key = 255

#Affin Parameters
Point_affin = [[int(0.5 * DISPLAY_W),int(0.5 * DISPLAY_H)], [int(0.8 * DISPLAY_W),int(0.2 * DISPLAY_H)], [int(0.8 * DISPLAY_W),int(0.8 * DISPLAY_H)], [int(0.2 * DISPLAY_W),int(0.8 * DISPLAY_H)]]

# Num of print grid
NUM_GRID = 5 
NUM_LIST = 20

# Eye gaze parameter
EYE_DEGREE_PARAMETER = DISTANCE / 10
TB_WEIGHT = 1.15

# User-Specific Measurements
USER_FACE_WIDTH = 140  # [mm]

# Configuration Parameters
PRINT_DATA = False
DEFAULT_WEBCAM = 0
SHOW_ALL_FEATURES = True
LOG_DATA = True
LOG_ALL_FEATURES = False
ENABLE_HEAD_POSE = True

# Blink Detection Parameters
SHOW_ON_SCREEN_DATA = True
TOTAL_BLINKS = 0
EYES_BLINK_FRAME_COUNTER = 0
BLINK_THRESHOLD = 0.51
EYE_AR_CONSEC_FRAMES = 2

# Head Pose Estimation Landmark Indices
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

# MediaPipe Model Confidence Parameters
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

# Angle Normalization Parameters
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
initial_pitch, initial_yaw, initial_roll = None, None, None
initial_diff_eye_x,initial_diff_eye_y = None,None
initial_x, initial_y = None, None
calibrated = False