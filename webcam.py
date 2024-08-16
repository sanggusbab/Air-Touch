import cv2

# 웹캠을 캡처 객체로 초기화 (기본적으로 0번 웹캠 사용)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 무한 루프: 프레임을 계속해서 읽어옴
while True:
    # 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 프레임을 창에 표시
    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
