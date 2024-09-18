import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import pyautogui
from click import HandClickDetector
from gaze import GazeEstimator


def main():
    import threading
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found")
        return
    hand_click_detector = HandClickDetector()
    Gaze = GazeEstimator()

    x, y = 0, 0
    is_Clicking_gesture = False

    lock = threading.Lock()

    frame = None
    frame_ready = threading.Event()
    stop_event = threading.Event()

    gazePeriod = 1
    gesturePeriod = 1
    windowPeriod = 5

    # Thread for capturing frames
    def frame_capture():
        nonlocal frame
        while not stop_event.is_set():
            ret, new_frame = cap.read()
            if not ret:
                stop_event.set()
                break
            with lock:
                frame = new_frame.copy()
            frame_ready.set()
            # Small sleep to prevent excessive CPU usage
            cv2.waitKey(1)

    # Thread for gaze estimation
    def gaze_thread():
        nonlocal x, y
        count = 0
        while not stop_event.is_set():
            frame_ready.wait()
            with lock:
                current_frame = frame.copy()
            if count % gazePeriod == 0:
                x_new, y_new = Gaze.gaze(current_frame)  # TODO: develop gaze module
                x_new = max(50, min(x_new, 1080 - 50))
                y_new = max(50, min(y_new, 1920 - 50))
                with lock:
                    x, y = x_new, y_new
            count += 1

    # Thread for gesture detection
    def gesture_thread():
        nonlocal is_Clicking_gesture
        count = 0
        while not stop_event.is_set():
            frame_ready.wait()
            with lock:
                current_frame = frame.copy()
            if count % gesturePeriod == 0:
                result = hand_click_detector.click(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
                if result:
                    with lock:
                        is_Clicking_gesture = True  # TODO: develop gesture module
            count += 1

    # Start the threads
    t_frame = threading.Thread(target=frame_capture)
    t_gaze = threading.Thread(target=gaze_thread)
    t_gesture = threading.Thread(target=gesture_thread)

    t_frame.start()
    t_gaze.start()
    t_gesture.start()

    cycleCounter = 0

    try:
        while not stop_event.is_set():
            cycleCounter += 1

            # Wait for the frame to be ready
            frame_ready.wait()
            frame_ready.clear()

            with lock:
                x_current, y_current = x, y
                click_gesture = is_Clicking_gesture

            if cycleCounter % windowPeriod == 0:
                if click_gesture:
                    print(f'You clicked {x_current}, {y_current}')
                    pyautogui.click(x_current, y_current)
                    with lock:
                        is_Clicking_gesture = False

            print(x_current, y_current)
            pyautogui.moveTo(x_current, y_current)

    except KeyboardInterrupt:
        stop_event.set()

    # Clean up
    stop_event.set()
    t_frame.join()
    t_gaze.join()
    t_gesture.join()
    cap.release()

if __name__ == '__main__':
    pyautogui.FAILSAFE = False
    main()
