#starter code to check opencv
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

path1 = '/Users/avinashkaur/Desktop/WindowShopper/data/frame1.jpg'


if os.path.isfile(path1):
        MIN_MATCH_COUNT = 10
        img1 = cv2.imread(path1)        #object to be detected

        gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

	print img1.shape



	'''
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray,None)

        #img=cv2.drawKeypoints(gray,kp)
        img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # plt.imshow(img),plt.show()
        print len(kp)
	'''



else:
        print("file does not exists")


