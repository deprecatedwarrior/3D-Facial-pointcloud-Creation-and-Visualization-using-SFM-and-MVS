#Using OpenCV takes a mp4 video and produces a number of images
import cv2
import numpy as np
import os
from scipy import ndimage, misc

def vid2frame():
	# Playing video from file:
	cap = cv2.VideoCapture('sampleVid1.MOV')
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	try:
    		if not os.path.exists('data'):
        		os.makedirs('data')
	except OSError:
	    	print('Error: Creating directory of data')

	currentFrame = 0
	while(currentFrame<length):
    
		# Capture frame-by-frame
    		ret, frame = cap.read()
		frame = ndimage.rotate(frame, 270)

		# Saves image of the current frame in jpg file
		name = './data/frame' + str(currentFrame) + '.jpg'
		print ('Creating...' + name)
		cv2.imwrite(name, frame)

		# To stop duplicate images
		currentFrame += 1

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
