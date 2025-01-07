import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cvzone
import cv2
import math
from ultralytics import YOLO

# ROS 초기화
rospy.init_node('fire_detection_node', anonymous=True)

# CvBridge 객체 생성
bridge = CvBridge()

# YOLO 모델 로드
model = YOLO('/home/yang/Desktop/family/fire_model.pt')

# 클래스 이름 정의
classnames = ['fire']

# 카메라 피드를 받을 콜백 함수
def image_callback(msg):
    # ROS 이미지를 OpenCV 이미지로 변환
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    # 프레임 크기 조정
    frame = cv2.resize(frame, (640, 480))
    
    # YOLO 모델 추론
    result = model(frame, stream=True)

    # 바운딩 박스, 신뢰도, 클래스 정보 처리
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 70:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
    
    # 화면에 결과 출력
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# 구독자 생성
rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

# ROS 루프 시작
rospy.spin()
