import numpy as np
import cv2 as cv
from constants import *

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
    cv.imshow("Eye Tracking", frame)
