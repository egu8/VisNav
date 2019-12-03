#!/usr/bin/env python
import rospy
from std_msgs.msg import String, UInt32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy

class turn_to_centroid:
	def __init__(self):
		self.pub = rospy.Publisher('/cmd_vel_mux/input/navi',Twist, queue_size = 10)
		self.msg = Twist()
		self.centroid_x = 320
		self.centroid_y = 240
		self.distance = 0
		self.bridge = CvBridge()
		self.move = rospy.Subscriber("/centroid", UInt32, self.callback)
		self.depth = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

	def depth_callback(self,data):
		depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
		cv2.imshow("depth camera", depth_image)
		self.distance = depth_image[self.centroid_x][self.centroid_y]
		rospy.loginfo(self.distance)
		cv2.waitKey(3)

	def callback(self, data):
		num = data.data >> 20
		self.centroid_x = data.data >> 10
		self.centroid_y = data.data & 0x1ff
		#rospy.loginfo('x: ' + str(self.centroid_x) + ' y: ' + str(self.centroid_y))
		if num > 0:
			self.msg.angular.z = 1
		else:
			lateral_distance = 320 - (self.centroid_x)
			if lateral_distance > 10:
				self.msg.angular.z = 0.2;
			elif lateral_distance < -10:
				self.msg.angular.z = -0.2;
			else:
				self.msg.angular.z = 0;
		self.msg.linear.x = 0.2
		if self.distance < 1.0:
			self.msg.linear.x = 0.0
		self.pub.publish(self.msg)

def listener():
	rospy.init_node('turn_to_centroid')
	thing = turn_to_centroid()
	rospy.spin()

if __name__ == '__main__':
	listener()

