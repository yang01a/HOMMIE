#!/usr/bin/env python3
import rospy
import random
import numpy as np
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
import actionlib

# 전역 변수로 맵 데이터 저장
map_data = None

def map_callback(data):
    global map_data
    map_data = data
    rospy.loginfo("Map data received. Processing map boundaries...")

def get_random_goal():
    global map_data

    if map_data is None:
        rospy.logwarn("Map data not yet received.")
        return None

    # 맵 정보 추출
    resolution = map_data.info.resolution
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y
    width = map_data.info.width
    height = map_data.info.height

    # OccupancyGrid 데이터를 numpy 배열로 변환
    grid = np.array(map_data.data).reshape((height, width))

    # 이동 가능한 영역(값: 0) 필터링
    free_cells = np.argwhere(grid == 0)

    if len(free_cells) == 0:
        rospy.logerr("No free cells found in the map!")
        return None

    # 랜덤으로 이동 가능한 셀 선택
    random_cell = free_cells[random.randint(0, len(free_cells) - 1)]

    # 셀의 좌표를 실제 좌표로 변환
    random_x = origin_x + random_cell[1] * resolution
    random_y = origin_y + random_cell[0] * resolution

    return random_x, random_y

def move_to_random_goal():
    global map_data

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"

    while not rospy.is_shutdown():
        # 랜덤 목표 지점 생성
        random_goal = get_random_goal()
        if random_goal is None:
            rospy.sleep(1)
            continue

        random_x, random_y = random_goal

        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = random_x
        goal.target_pose.pose.position.y = random_y
        goal.target_pose.pose.orientation.w = 1.0

        rospy.loginfo(f"Moving to goal: x={random_x}, y={random_y}")
        client.send_goal(goal)
        client.wait_for_result()

        rospy.sleep(2)

if __name__ == "__main__":
    try:
        rospy.init_node('random_patrol')

        # 맵 데이터 구독
        rospy.Subscriber('/map', OccupancyGrid, map_callback)

        move_to_random_goal()
    except rospy.ROSInterruptException:
        pass

