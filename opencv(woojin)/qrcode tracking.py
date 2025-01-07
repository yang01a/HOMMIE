#!/usr/bin/env python3
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import re
import signal
import time

# 시그널 처리 함수 (Ctrl+C 또는 종료 시 호출)
def signal_handler(sig, frame):
    rospy.signal_shutdown("Signal received")  # ROS 노드를 종료합니다.

# QR 코드 데이터 파싱 함수
def parse_qr_data(data):
    # 정규 표현식을 사용하여 QR 코드에서 x, y, theta 값 추출
    pattern = r"x:(-?\d+\.?\d*), y:(-?\d+\.?\d*), theta:(-?\d+\.?\d*)"
    match = re.search(pattern, data)  # 정규 표현식으로 데이터 매칭
    if match:
        # 매칭된 값들을 실수형으로 변환
        x = float(match.group(1))
        y = float(match.group(2))
        theta = float(match.group(3))
        return x, y, theta
    else:
        rospy.logwarn("Invalid QR code data format")  # QR 코드 데이터 형식이 잘못되었을 때 경고
        return None, None, None

# 로봇 목표 위치로 이동하는 함수
def move_to_goal(x, y, theta):
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)  # 목표 위치 퍼블리셔
    rospy.sleep(1)  # 퍼블리셔가 준비될 때까지 1초 대기

    goal = PoseStamped()  # 목표 위치 메시지 생성
    goal.header.frame_id = "map"  # 목표 위치가 "map" 좌표계 기준임을 명시
    goal.header.stamp = rospy.Time.now()  # 현재 시간으로 타임스탬프 설정

    # 목표 위치 설정
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.z = np.sin(theta / 2.0)  # 회전 각도를 쿼터니언으로 변환
    goal.pose.orientation.w = np.cos(theta / 2.0)

    rospy.loginfo(f"Sending goal: x={x}, y={y}, theta={theta}")  # 목표 위치 로깅
    goal_pub.publish(goal)  # 목표 위치 퍼블리시

# QR 코드 탐지 및 트래킹 함수
def qr_code_detection_with_tracking():
    rospy.init_node('qr_code_detection', anonymous=True)  # ROS 노드 초기화
    qr_pub = rospy.Publisher('/qr_code_data', String, queue_size=10)  # QR 코드 데이터를 퍼블리시하는 퍼블리셔
    rospy.loginfo("QR Code Detection with Tracking Node Started")  # 노드 시작 메시지

    cap = cv2.VideoCapture(0)  # 웹캠에서 비디오 캡처 시작
    if not cap.isOpened():
        rospy.logerr("Cannot open camera.")  # 카메라 열기 실패 시 에러 메시지
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 카메라 해상도 설정 (너비)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 카메라 해상도 설정 (높이)

    tracker = None  # 트래커 초기화
    tracking_active = False  # 트래킹 활성화 여부
    prev_time = time.time()  # 이전 프레임 시간 초기화

    while not rospy.is_shutdown():
        ret, frame = cap.read()  # 카메라에서 프레임을 읽어옴
        if not ret:
            rospy.logwarn("Cannot read frame from camera.")  # 프레임 읽기 실패 시 경고
            continue

        cur_time = time.time()  # 현재 시간
        fps = 1 / (cur_time - prev_time)  # FPS 계산
        prev_time = cur_time  # 이전 시간 갱신
        rospy.loginfo(f"FPS: {fps:.2f}")  # FPS 로깅

        # QR 코드 탐지 또는 트래킹
        if not tracking_active:  # 트래킹이 활성화되지 않은 경우
            decoded_objects = decode(frame)  # QR 코드 디코딩
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')  # QR 코드에서 데이터를 UTF-8로 디코딩
                rospy.loginfo(f"QR Code Detected: {data}")  # 디텍션된 QR 코드 데이터 로깅
                qr_pub.publish(data)  # QR 코드 데이터를 퍼블리시

                # QR 코드에서 좌표 파싱
                x, y, theta = parse_qr_data(data)
                if x is not None and y is not None:  # 유효한 좌표가 있을 경우
                    move_to_goal(x, y, theta)  # 목표 위치로 이동
                    tracking_active = True  # 트래킹 활성화

                # 트래커 초기화 (QR 코드 영역 설정)
                points = np.array([(p.x, p.y) for p in obj.polygon])  # QR 코드의 다각형 모서리 점들
                x_min, y_min = points.min(axis=0)  # 최소 좌표 (왼쪽 상단)
                x_max, y_max = points.max(axis=0)  # 최대 좌표 (오른쪽 하단)

                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)  # 경계 상자 정의
                tracker = cv2.TrackerKCF_create()  # KCF 트래커 생성
                tracking_active = tracker.init(frame, bbox)  # 트래커 초기화
                rospy.loginfo("Tracking Initialized")  # 트래킹 초기화 완료 로깅
                break
        else:
            success, bbox = tracker.update(frame)  # 트래킹 수행
            if success:  # 트래킹 성공 시
                x, y, w, h = [int(v) for v in bbox]  # 트래킹된 객체의 위치와 크기
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형으로 트래킹 영역 표시
                cv2.putText(frame, "Tracking QR Code", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # "Tracking QR Code" 텍스트 삽입
            else:  # 트래킹 실패 시
                rospy.logwarn("Tracking Lost")  # 트래킹 실패 경고
                tracking_active = False  # 트래킹 비활성화

        # 상태 표시
        status = "Tracking" if tracking_active else "Searching"  # 트래킹 상태 표시
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 상태 텍스트 화면에 출력
        cv2.imshow("QR Code Detection and Tracking", frame)  # 프레임을 화면에 출력

        # ESC 키가 눌리면 종료
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 눌렀을 때
            rospy.signal_shutdown("ESC key pressed")  # ROS 노드 종료
            break

    cap.release()  # 카메라 해제
    cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 종료

# main 함수
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # SIGINT (Ctrl+C) 시그널 처리 함수 연결
    try:
        qr_code_detection_with_tracking()  # QR 코드 탐지 및 트래킹 함수 호출
    except rospy.ROSInterruptException:
        pass  # ROS 인터럽트 예외 처리
