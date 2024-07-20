import cv2
import numpy as np
import sys

# cascade classifiers for face and eyes(다운필수)
face_cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('.\haarcascade_eye.xml')

video = '.\eyetrack.mp4'
cap = cv2.VideoCapture(0)

'''def blob_process(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 42, 255, cv2.THRESH_BINARY)
    keypoints = detector.detect(img)
    return keypoints '''

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #cut
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                #print(x, y, w, h)
                cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0, 0),2) 
                #흑백으로 할 거라 gray라 씀/컬러로 할 거면 frame 쓰기
                height = int(h / 2)
                width = int(w/2)
                gray_face = gray[y:y+ height, x:x+width] #왼쪽 눈만 남김
                eyes = eye_cascade.detectMultiScale(gray_face)

                for (ex,ey,ew,eh) in eyes: #이거 잘 안 됨.. 얼굴 위로 어케 자르냐 -> 해결
                    cv2.rectangle(gray_face,(ex,ey),(ex+ew,ey+eh),(255, 0, 255),2)
                    # eye_width = np.size(gray_eye, 1) /2
                    # eye_height = np.size(gray_eye, 0)/2
                    #eye_center = eye_width/2, eye_height/2
                    xx = int(ex + (0.5 * ew))
                    yy= int(ey + (0.5 * eh))
                    print(f"x: {xx}\ty: {yy}")
                    cv2.circle(gray_face,(xx, yy), 5,(255, 0, 0),thickness=2)
                    gray_eye = gray_face[ey:ey+eh, ex:ex+ew]
                    cv2.imshow("eye", gray_eye)
                    cv2.imshow("original", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'): 
                break
        else:
            break
else:
    print('cannot open the file')

cap.release()
cv2.destroyAllWindows()
