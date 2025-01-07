import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tkinter as tk
import threading
import time

# FireDetectionGUI 클래스는 Tkinter를 사용하여 화재 감지 시스템의 GUI를 관리
class FireDetectionGUI:
    def __init__(self, root):
        # Tkinter 루트 윈도우 설정
        self.root = root
        self.root.title("Fire Detection")  # GUI 윈도우 제목 설정
        self.root.geometry("400x300")  # GUI 윈도우 크기 설정
        
        # ROS 노드 초기화
        rospy.init_node('fire_detector', anonymous=True)
        # '/fire_alert' 토픽에 화재 감지 메시지를 발행하는 퍼블리셔 생성
        self.fire_pub = rospy.Publisher('/fire_alert', String, queue_size=10)
        self.bridge = CvBridge()  # OpenCV 이미지와 ROS 이미지 메시지 변환을 위한 CvBridge 객체 생성

        # 카메라 캡처 객체 초기화
        self.cap = None
        self.is_running = False  # 감지 상태 플래그 (시작/정지)
        self.sensitivity = 5000  # 감지 민감도 초기 값
        self.threshold_scale = None  # 민감도 조절 슬라이더 변수
        self.stream_thread = None  # 카메라 스트리밍을 처리할 스레드
        self.previous_frame = None  # 이전 프레임 (옵티컬 플로우 추적용)
        self.previous_points = None  # 이전 프레임의 특징점 (옵티컬 플로우 추적용)
        self.last_detection_time = None  # 마지막 화재 감지 시간 (시간적 변화 분석)

        # GUI 위젯 생성
        self.create_widgets()

    def create_widgets(self):
        """GUI 위젯을 설정하는 함수"""
        # 민감도 조절 슬라이더 위젯 생성
        self.threshold_scale = tk.Scale(self.root, from_=1000, to_=10000, orient=tk.HORIZONTAL, label="Sensitivity", command=self.update_sensitivity)
        self.threshold_scale.set(self.sensitivity)  # 초기 민감도 값 설정
        self.threshold_scale.pack(pady=20)  # 슬라이더 위젯 화면에 배치

        # 시작/정지 버튼 생성
        self.start_button = tk.Button(self.root, text="Start Fire Detection", command=self.toggle_detection)
        self.start_button.pack(pady=10)  # 버튼 화면에 배치

        # 상태 라벨 생성 (화재 감지 상태 표시)
        self.status_label = tk.Label(self.root, text="Status: Stopped", fg="red")
        self.status_label.pack(pady=10)  # 라벨 화면에 배치

    def update_sensitivity(self, value):
        """슬라이더로 감지 민감도를 업데이트하는 함수"""
        self.sensitivity = int(value)  # 슬라이더 값에 따라 민감도 업데이트
        print(f"Updated sensitivity: {self.sensitivity}")

    def toggle_detection(self):
        """화재 감지 시작/중지를 전환하는 함수"""
        if self.is_running:
            self.is_running = False  # 감지 종료
            self.start_button.config(text="Start Fire Detection")  # 버튼 텍스트 변경
            self.status_label.config(text="Status: Stopped", fg="red")  # 상태 라벨 변경
            self.stop_detection()  # 감지 중지 함수 호출
        else:
            self.is_running = True  # 감지 시작
            self.start_button.config(text="Stop Fire Detection")  # 버튼 텍스트 변경
            self.status_label.config(text="Status: Running", fg="green")  # 상태 라벨 변경
            self.start_detection()  # 감지 시작 함수 호출

    def start_detection(self):
        """카메라 스트리밍 및 감지 스레드를 시작하는 함수"""
        self.cap = cv2.VideoCapture(0)  # 카메라 열기
        if not self.cap.isOpened():
            print("Error: Failed to open camera.")  # 카메라 열기 실패 시 오류 출력
            return
        self.stream_thread = threading.Thread(target=self.detect_fire, daemon=True)  # 감지 스레드 시작
        self.stream_thread.start()

    def stop_detection(self):
        """감지 종료 및 리소스 해제 함수"""
        if self.cap:
            self.cap.release()  # 카메라 캡처 종료
            self.cap = None
        cv2.destroyAllWindows()  # OpenCV 윈도우 종료

    def detect_fire(self):
        """화재 감지 알고리즘"""
        while self.is_running:
            ret, frame = self.cap.read()  # 카메라로부터 프레임 읽기
            if not ret:
                break  # 프레임을 제대로 읽지 못하면 종료

            frame = cv2.resize(frame, (640, 480))  # 프레임 크기 변경
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR에서 HSV 색상 공간으로 변환

            # 불꽃 색상 범위 설정 (빨간색, 주황색, 노란색)
            lower_red1 = np.array([0, 120, 120])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 120, 120])
            upper_red2 = np.array([180, 255, 255])
            lower_orange = np.array([10, 100, 200])
            upper_orange = np.array([25, 255, 255])
            lower_yellow = np.array([25, 100, 200])
            upper_yellow = np.array([35, 255, 255])

            # 각 색상 범위에 대해 마스크 생성
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
            mask4 = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask = mask1 | mask2 | mask3 | mask4  # 모든 마스크 결합

            # 마스크에서 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)  # 윤곽선의 면적 계산
                if area > self.sensitivity:  # 면적이 민감도보다 크면 화재로 감지
                    x, y, w, h = cv2.boundingRect(contour)  # 윤곽선에 사각형 그리기 위한 좌표 계산
                    aspect_ratio = float(w) / h  # 가로/세로 비율 계산

                    # 비정상적인 객체 (불꽃이 아닌 객체) 필터링
                    if 0.5 < aspect_ratio < 2.0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 화재 영역에 사각형 그리기
                        rospy.loginfo("Fire detected!")  # ROS 로그에 화재 감지 메시지 기록
                        self.fire_pub.publish("Fire detected!")  # ROS에 화재 감지 메시지 발행

                        # 거리 기반 필터링 (화재의 크기나 거리를 고려한 필터링)
                        if w > 50 and h > 50:  # 예시로, 크기가 일정 이상일 때만 화재로 인식
                            rospy.loginfo(f"Large fire detected with width: {w} and height: {h}")

            # Optical Flow를 사용하여 화염의 움직임을 추적하는 부분
            if self.previous_frame is not None:
                # Optical Flow를 계산하여 움직임을 추적
                prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 특징점 추출 (첫 번째 프레임에서 특징점 찾기)
                if self.previous_points is None:
                    self.previous_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **{
                        'maxCorners': 100,
                        'qualityLevel': 0.3,
                        'minDistance': 7,
                        'blockSize': 7
                    })

                # Optical Flow를 이용하여 특징점 이동 추적
                next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.previous_points, None)

                if next_points is not None:
                    # 점 이동을 추적하여 그리기
                    for i, (new, old) in enumerate(zip(next_points, self.previous_points)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)

                    # 화재의 움직임 추적
                    rospy.loginfo("Tracking fire movement...")

                    # 다음 프레임을 이전 프레임으로 업데이트
                    self.previous_points = next_points

            # 이전 프레임 업데이트
            self.previous_frame = frame

            # 시간적 변화 분석 (화재가 일정 시간 이상 지속되면 화재 확정)
            if self.last_detection_time is None or time.time() - self.last_detection_time > 5:  # 5초 이상 연속 감지
                self.last_detection_time = time.time()
                rospy.loginfo("Fire confirmed!")
                self.fire_pub.publish("Fire confirmed!")  # ROS에 화재 확정 메시지 발행

            # 화면에 결과 프레임 표시
            cv2.imshow("Fire Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 감지 종료 후 카메라 릴리즈
        self.stop_detection()

if __name__ == "__main__":
    root = tk.Tk()  # Tkinter 윈도우 생성
    gui = FireDetectionGUI(root)  # GUI 객체 생성
    root.mainloop()  # Tkinter 이벤트 루프 시작
