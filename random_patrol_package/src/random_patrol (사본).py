#!/usr/bin/env python3
import rospy
import random
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib

def move_to_random_goal():
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    while not rospy.is_shutdown():
        random_x = random.uniform(-5, 5)
        random_y = random.uniform(-5, 5)

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
        move_to_random_goal()
    except rospy.ROSInterruptException:
        pass

