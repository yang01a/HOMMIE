#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import random
import string
from std_msgs.msg import String

def generate_random_qr():
    """랜덤한 QR 코드를 생성하는 함수"""
    # 랜덤 문자열 생성 (여기서는 10자리 랜덤 문자열)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    rospy.loginfo(f"Generated QR Code: {random_str}")
    return random_str

def qr_publisher():
    """QR 코드 정보를 퍼블리시하는 노드"""
    rospy.init_node('random_qr_generator', anonymous=True)
    pub = rospy.Publisher('random_qr_code', String, queue_size=10)
    rate = rospy.Rate(1)  # 1Hz 주기로 퍼블리시
    
    while not rospy.is_shutdown():
        qr_code = generate_random_qr()
        pub.publish(qr_code)
        rate.sleep()

if __name__ == '__main__':
    try:
        qr_publisher()
    except rospy.ROSInterruptException:
        pass

