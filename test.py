import numpy as np
import cv2 as cv
import pyautogui
import time
import statistics
from constants import *
from eye_test import GazeEstimator

# Parameters for moving dot
DOT_RADIUS = 10
DOT_SPEED = 8
DISPLAY_W, DISPLAY_H = 2900, 1600

class MovingDotTracker:
    def __init__(self):
        self.dot_position = [DISPLAY_W // 2, DISPLAY_H // 2]
        self.dot_direction = [1, 1]  # Moving diagonally initially
        self.mouse_errors = []
        self.gaze_errors = []

    def update_dot_position(self):
        # Update the dot position
        for i in range(2):
            self.dot_position[i] += DOT_SPEED * self.dot_direction[i]
            # Reverse direction if hitting the border
            if self.dot_position[i] <= DOT_RADIUS or self.dot_position[i] >= (DISPLAY_W if i == 0 else DISPLAY_H) - DOT_RADIUS:
                self.dot_direction[i] *= -1

    def calculate_error(self, dot_position, cursor_position):
        # Calculate the distance between dot and cursor/gaze
        error = np.linalg.norm(np.array(dot_position) - np.array(cursor_position))
        return error

    def track_with_mouse(self):
        # Reset dot position to the center of the screen
        self.dot_position = [DISPLAY_W // 2, DISPLAY_H // 2]
        cap = cv.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Update the dot position
                self.update_dot_position()

                # Get the current mouse position
                mouse_x, mouse_y = pyautogui.position()
                mouse_x -= 7
                mouse_y -= 55
                

                # Calculate error between dot and mouse cursor
                mouse_error = self.calculate_error(self.dot_position, [mouse_x, mouse_y])
                self.mouse_errors.append(mouse_error)

                # Draw moving dot on the frame
                frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)  # Black background
                cv.circle(frame, tuple(self.dot_position), DOT_RADIUS, (0, 255, 0), -1)
                cv.circle(frame, (mouse_x, mouse_y), 5, (255, 0, 0), -1)  # Draw mouse cursor for visualization

                # Show the frame
                cv.imshow('Mouse Tracking', frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv.destroyAllWindows()

        finally:
            cap.release()
            cv.destroyAllWindows()
            # Calculate average error
            if self.mouse_errors:
                average_mouse_error = statistics.mean(self.mouse_errors)
                print(f'Average Mouse Tracking Error: {average_mouse_error:.2f} pixels')

    def track_with_gaze(self, gaze_estimator):
        # Calibration loop until 's' key is pressed
        cap = cv.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)  # Black background
                cv.putText(frame, "Calibrating... Press 'k' to start tracking", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                # Assuming gaze_estimator is calibrated to screen center
                cv.imshow('Calibration', frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('k'):
                    break
                elif key == ord('r'):
                    gaze_estimator.reset()
        finally:
            cap.release()
            cv.destroyAllWindows()
        
        # Reset dot position to the center of the screen
        # Reset dot position to the center of the screen
        self.dot_position = [DISPLAY_W // 2, DISPLAY_H // 2]
        cap = cv.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Update the dot position
                self.update_dot_position()

                # Get the current gaze position using the GazeEstimator
                gaze_x, gaze_y = gaze_estimator.gaze(frame)

                # Calculate error between dot and gaze point
                gaze_error = self.calculate_error(self.dot_position, [gaze_x, gaze_y])
                self.gaze_errors.append(gaze_error)

                # Draw moving dot and gaze point on the frame
                frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)  # Black background
                cv.circle(frame, tuple(self.dot_position), DOT_RADIUS, (0, 255, 0), -1)
                if gaze_x is not None and gaze_y is not None:
                    cv.circle(frame, (gaze_x, gaze_y), 5, (255, 0, 0), -1)  # Draw gaze point for visualization

                # Show the frame
                cv.imshow('Gaze Tracking', frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            cap.release()
            cv.destroyAllWindows()
            # Calculate average error
            if self.gaze_errors:
                average_gaze_error = statistics.mean(self.gaze_errors)
                print(f'Average Gaze Tracking Error: {average_gaze_error:.2f} pixels')

if __name__ == '__main__':
    tracker = MovingDotTracker()
    
    # Step 1: Track with mouse
    print("Tracking with mouse. Press 'q' to quit.")
    tracker.track_with_mouse()

    # Step 2: Track with gaze estimator
    print("Tracking with gaze. Press 'q' to quit.")
    gaze_estimator = GazeEstimator()
    gaze_estimator.reset()  # Initial reset to calibrate to the center of the screen
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)  # Black background
        cv.putText(frame, "Calibrating... Press 'k' to start tracking", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('Calibration', frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('k'):
            break
    
    while True:
        # Perform repeated resets to ensure calibration is accurate before starting
        gaze_estimator.reset()
        key = cv.waitKey(1) & 0xFF
        if key == ord('k'):
            break
    tracker.track_with_gaze(gaze_estimator)
    
    cap.release()
    cv.destroyAllWindows()
