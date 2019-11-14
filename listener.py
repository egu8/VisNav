#!/usr/bin/env python
import rospy
import tf
import math
from geometry_msgs.msg import Twist

if __name__ == '__main__':
	rospy.init_node("return_to_origin")
	listener = tf.TransformListener()

	vel = rospy.Publisher('/cmd_vel_mux/input/navi',Twist, queue_size = 1)

	rate = rospy.Rate(1.0)

	while not rospy.is_shutdown():
		(trans,rot) = listener.lookupTransform('base_footprint','odom', rospy.Time(0))
		angular = 4 * math.atan2(trans[1], trans[0])
        linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2)
        cmd = Twist()
        rospy.loginfo(cmd)
        cmd.linear.x = linear
        cmd.angular.z = angular

        vel.publish(cmd)

        rate.sleep()