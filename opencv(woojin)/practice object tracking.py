import cv2
import rospy
from geometry_msgs.msg import Twist

# ROS 초기화
rospy.init_node('user_following')
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# 웹캠 연결
cap = cv2.VideoCapture(0)

# CSRT 추적기 초기화
tracker = cv2.TrackerCSRT_create()
initBB = None  # 초기 객체 위치

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전 (사용자 시야 기준)
    (H, W) = frame.shape[:2]

    if initBB is not None:
        # 객체 추적
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            center_x = x + w // 2
            center_y = y + h // 2

            # 시각화
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # ROS 제어 메시지
            twist = Twist()
            twist.linear.x = 0.1 if w < W // 3 else 0.0
            twist.angular.z = -0.002 * (center_x - W // 2)
            cmd_vel_pub.publish(twist)

    # 객체 초기 선택
    cv2.putText(frame, "Press 's' to select target", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False)
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
