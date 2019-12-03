#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String, UInt32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot
import numpy

class centroid_detector:

	def __init__(self):
		self.centroid_pub = rospy.Publisher("/centroid", UInt32, queue_size = 10)
		self.bridge = CvBridge()
		self.coord = 0x1fffff
		self.depth_image = None
		# self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
		self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)

	# def depth_callback(self, data):
	# 	 self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
	# 	 cv2.imshow("depth camera", self.depth_image)
	# 	 cv2.waitKey(3)

	def callback(self, data):
		image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		cv2.waitKey(3)
		(rows,cols,channels) = image.shape
		lower = numpy.array([50, 0, 50], dtype = "uint8")
		upper = numpy.array([255, 40, 255], dtype = "uint8")
		mask = cv2.inRange(image, lower, upper)
		thing = numpy.sum(mask)
		if thing > 20:
			M = cv2.moments(mask)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			#rospy.loginfo(cX)
 			cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
 			self.coord = ((cX << 10) + cY) & 0xfffff
 		else:
 			self.coord = 0x1fffff
		cv2.imshow("rgb", image)
		self.centroid_pub.publish(self.coord)

if __name__ == '__main__':
	cd = centroid_detector()
	rospy.init_node('centroid_detector', anonymous = False)
	rospy.spin()
	cv2.destroyAllWindows()