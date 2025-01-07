import cv2

class FaceDetector:
    def __init__(self):
        # Haarcascade 파일 경로를 직접 지정
        face_cascade_path = '/home/opulent/tf_env/OpenCV_python/haar-cascade-files-master/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # 카메라 실행 및 해상도 조정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # 가로 해상도 640
        self.cap.set(4, 480)  # 세로 해상도 480
        if not self.cap.isOpened():
            print("Camera initialization failed.")

    def preprocess_frame(self, img):
        """
        입력 프레임을 전처리하여 조명 효과를 줄이고 명암비를 개선합니다.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
        # GaussianBlur 적용하여 노이즈 감소
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def detect_faces(self, img):
        """
        얼굴을 검출하고, 이미지에 얼굴을 표시합니다.
        """
        # 프레임 전처리
        gray = self.preprocess_frame(img)

        # 얼굴 검출
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,  # minNeighbors 값을 5로 낮춰서 더 민감하게 설정
            minSize=(30, 30),
            maxSize=(500, 500)  # 너무 큰 얼굴은 제외
        )

        # 얼굴 영역에 사각형 그리고 텍스트 추가
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'human', (x, y - 10), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

        return img, faces

    def run(self):
        while True:
            ret, img = self.cap.read()
            if not ret:
                print("Camera frame not available.")
                continue

            # 얼굴 검출
            img, faces = self.detect_faces(img)

            # 영상 출력
            cv2.imshow('Face Detection', img)

            # ESC 키로 종료
            if cv2.waitKey(30) & 0xFF == 27:  
                break

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()
    detector.cleanup()
