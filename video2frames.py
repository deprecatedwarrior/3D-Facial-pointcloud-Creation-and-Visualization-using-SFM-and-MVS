#Using OpenCV takes a mp4 video and produces a number of images
import cv2
import numpy as np
import os
from scipy import ndimage, misc

class Video2Frames:
	def __init__(self, video):
		self.video = video
		
		
	def getFrames(self):
		# Playing video from file:
		cap = cv2.VideoCapture(self.video)
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		try:
			if not os.path.exists('data'):
				os.makedirs('data')
		except OSError:
			print('Error: Creating directory of data')

		currentFrame = 0
		while(currentFrame<length):
			ret, frame = cap.read()
			#frame = ndimage.rotate(frame, 270) # for this test video
			name = './data/frame' + str(currentFrame) + '.jpg'
			print ('Creating...' + name)
			cv2.imwrite(name, frame)
			currentFrame += 1

		# release capture when done
		cap.release()
		cv2.destroyAllWindows()
