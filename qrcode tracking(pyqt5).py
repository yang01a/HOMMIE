import sys
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import rospy
from std_msgs.msg import String

class QRCodeROSApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # ROS 초기화
        rospy.init_node('qr_code_ros_node', anonymous=True)
        self.qr_pub = rospy.Publisher('/qr_code_data', String, queue_size=10)

        # GUI 초기화
        self.setWindowTitle("QR Code Detection with ROS")
        self.setGeometry(100, 100, 800, 600)

        # 상태 변수
        self.detection_active = False

        # UI 요소 생성
        self.start_button = QPushButton("Start Detection", self)
        self.start_button.clicked.connect(self.toggle_detection)

        self.status_label = QLabel("Status: Idle", self)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Error: Unable to access the camera")

        # 타이머 시작
        self.timer.start(30)  # 30ms마다 실행

    def toggle_detection(self):
        """QR 코드 감지 ON/OFF 전환"""
        self.detection_active = not self.detection_active
        if self.detection_active:
            self.status_label.setText("Status: Detecting QR Codes")
            self.start_button.setText("Stop Detection")
        else:
            self.status_label.setText("Status: Streaming Only")
            self.start_button.setText("Start Detection")

    def process_frame(self):
        """QR 코드 감지 및 스트리밍 처리"""
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Error: Unable to read frame")
            return

        if self.detection_active:
            # 이미지 전처리 (그레이스케일, 대비 조정)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.equalizeHist(gray_frame)  # 대비 조정

            # QR 코드 디코딩
            decoded_objects = decode(frame)

            # ROS 메시지 송신
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                rospy.loginfo(f"QR Code Detected: {data}")
                self.qr_pub.publish(data)

            # QR 코드 영역 표시
            frame = self.draw_corrected_qr(frame, decoded_objects)

        # 화면 출력
        cv2.imshow("QR Code Detection", frame)

    def draw_corrected_qr(self, frame, decoded_objects):
        """QR 코드 영역을 원근 보정하여 표시"""
        for obj in decoded_objects:
            points = np.array([point for point in obj.polygon], dtype=np.float32)

            if len(points) != 4:
                points = cv2.convexHull(points)

            # QR 코드의 왜곡을 보정하기 위해 원근 변환
            width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
            dst_points = np.array([
                [0, 0],
                [width, 0],
                [width, width],
                [0, width]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(points, dst_points)
            warped = cv2.warpPerspective(frame, matrix, (int(width), int(width)))

            # 원근 보정된 QR 코드 이미지 출력
            cv2.imshow("Warped QR Code", warped)

            # 원본 이미지에 QR 코드의 경계 그리기
            points = points.astype(np.int32)
            for i in range(len(points)):
                cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 2)

        return frame

    def closeEvent(self, event):
        """종료 시 자원 해제"""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QRCodeROSApp()
    window.show()
    sys.exit(app.exec_())
