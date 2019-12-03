#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String, UInt32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot
import numpy as np
import torch
import torchvision.models as models

# Takes a predefinied model and gives predictions
def predict(x):

	model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	model.eval()
	predictions = model(x)
	
	return predictions

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

		torch_image = np.copy(image)
		torch_image = np.transpose(torch_image, (2,0,1))
		torch_image = torch.from_numpy(torch_image)
		torch_image = torch_image.type(torch.FloatTensor)
		torch_image/= 255
		torch_image = [torch_image]


		predictions = predict(torch_image)
		predictions = predictions[0]

		print("FOWARD PASS SUCCESS")

		boxes = predictions['boxes']
		scores = predictions['scores']



		for i in range(len(scores)):
			if scores[i] < 0.7:
				continue
			else:
				box = boxes[i]

				x1, y1, x2, y2 = box

				cv2.rectangle(image, (x1,y1), (x2,y2), (255, 255, 255))

		cv2.imshow("rgb", image)
		cv2.waitKey(3)
		self.centroid_pub.publish(self.coord)
		print("DONE")

if __name__ == '__main__':
	cd = centroid_detector()
	rospy.init_node('object_detector', anonymous = False)
	rospy.spin()
	cv2.destroyAllWindows()



