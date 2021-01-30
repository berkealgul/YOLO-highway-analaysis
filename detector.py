import time
import sys
import torch as T
import numpy as np
import cv2

try:
	from Yolo.utils import *
	from Yolo.darknet import *
except:
	from utils import *
	from darknet import *


class Detector:
	def __init__(self, config="config.cfg", weights="weights/yolov3.weights", 
					classes="data/coco.names"):
		self.yolo = Darknet(config, weights)
		self.CUDA = T.cuda.is_available()
		self.num_classes = 80
		self.classes = load_classes(classes)
		self.colors = create_colors(self.num_classes)
		self.in_dim = 416
		self.valid_classes = [2, 5, 7]

		if self.CUDA:
			self.yolo.cuda()

		self.yolo.eval()
		self.recentCoordinates = list()

		print("CUDA: ", self.CUDA)
		
	def detect(self, frame):
		img = prep_image(frame.copy() ,self.in_dim)

		if self.CUDA:
			img = img.cuda()

		result = self.yolo(img, self.CUDA)
		result = adjust_results(result, 0.5, self.num_classes)

		try:
			coordinates = write_result(result.clone(), frame, self.in_dim, 
				self.classes, self.colors, self.valid_classes)
			
			return coordinates
		except: 
			return None


if __name__ == "__main__":
	cap = cv2.VideoCapture("video.mp4")
	d = Detector()
	
	while True:
		ret, frame = cap.read()

		if not ret:
			break
		
		print(d.detect(frame))
		
		"""
        cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color,-1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
		"""
		cv2.imshow("frame", frame)
		cv2.waitKey(1)
	