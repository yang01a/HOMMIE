#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String

def spawn_station():
    rospy.init_node('spawn_station_with_qr')

    pub = rospy.Publisher('/station_pose', Pose, queue_size=10)
    rate = rospy.Rate(10)

    # 예시로 특정 위치에 QR 코드와 함께 스테이션을 생성
    pose = Pose()
    pose.position.x = -1.0
    pose.position.y = 2.0
    pose.position.z = 0.05
    pose.orientation.w = 1.0

    # QR 코드 정보 (예시)
    qr_code = "QR CODE INFO: Station -1,2"

    rospy.loginfo("Spawning station with QR code at: (%f, %f, %f)", pose.position.x, pose.position.y, pose.position.z)
    pub.publish(pose)

    rospy.loginfo("QR Code: %s", qr_code)

    rospy.spin()

if __name__ == '__main__':
    try:
        spawn_station()
    except rospy.ROSInterruptException:
        pass

