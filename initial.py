import cv2 as cv
import numpy as np
from constants import *
import constants

def Affin(frame, gaze):
    
    global SETUP_STEP, key, gaze_points
    h, w, c = frame.shape
    setup_D = np.zeros((DISPLAY_H, DISPLAY_W), np.uint8)
    cv.namedWindow('setup_page', cv.WINDOW_NORMAL)
    cv.setWindowProperty('setup_page', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    if(constants.SETUP_STEP == 0) :
        gaze_points = [[0,0],[0,0],[0,0]]
        # print(Point_affin)
        # print(int(Point_affin[0][0]),int(Point_affin[0][1]))
        cv.circle(setup_D, (int(Point_affin[0][0]),int(Point_affin[0][1])), 10, (255, 0, 0), 5,cv.LINE_AA)
        cv.imshow('setup_page',setup_D)
        # print(2)
        # cv.putText(setup_D, f"Please look at this Dot and press button", frame_A, cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        
    elif(constants.SETUP_STEP == 1) :
        cv.circle(setup_D, (int(Point_affin[1][0]),int(Point_affin[1][1])), 10, (255, 0, 0), 5, cv.LINE_AA)
        cv.imshow('setup_page',setup_D)
    
    elif(constants.SETUP_STEP == 2) :
        if(gaze_points[0] == [0,0]):
            gaze_points[0] = gaze
            print("STEP_TWO clear")
        cv.circle(setup_D, (int(Point_affin[2][0]),int(Point_affin[2][1])), 10, (255, 0, 0), 5, cv.LINE_AA)
        cv.imshow('setup_page',setup_D)
        
    elif(constants.SETUP_STEP == 3) :
        if(gaze_points[1] == [0,0]):
            gaze_points[1] = gaze
            print("STEP_THREE clear")
        cv.circle(setup_D, (int(Point_affin[3][0]),int(Point_affin[3][1])), 10, (255, 0, 0), 5, cv.LINE_AA)
        cv.imshow('setup_page',setup_D)
        
    elif(constants.SETUP_STEP == 4) :
        if(gaze_points[2] == [0,0]):
            gaze_points[2] = gaze
            print("STEP_FOUR clear")
            cv.destroyWindow( 'setup_page' )
            print(Point_affin[1:4], gaze_points)
        
            A = np.float32()
            Affin_Matrix = cv.getAffineTransform(np.float32(Point_affin[1:3]), np.float32(gaze_points))
            screw_frame = cv.warpAffine(frame, Affin_Matrix, (w,h))
    
            return screw_frame

    return frame

