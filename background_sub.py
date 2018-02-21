import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from segmentation import edgeDetect, findSignificantContours
from scipy.signal import savgol_filter


def preprocessing(filename): # Image resizing is performed here

	 #Read image
        img = cv2.imread(filename)
	
	#Gaussian Blurring for noise removal
        blurred = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise

        #Conversion for display
        img_show = cv2.cvtColor(blurred,cv2.COLOR_BGR2RGB)

        #gray conversion
        gray= cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)

        #Since we're operating on 3 channel images, we perform this on all channels and take out the maximum 'edge value' from all three
        edgeImg = np.max( np.array([ edgeDetect(img_show[:,:, 0]), edgeDetect(img_show[:,:, 1]), edgeDetect(img_show[:,:, 2]) ]), axis=0 )

        # Zero any value that is less than mean of the edge Image. This reduces a lot of noise.
        mean = np.mean(edgeImg);
        edgeImg[edgeImg <= mean] = 0;

        edgeImg_8u = np.asarray(edgeImg, np.uint8)

        # Find contours
        significant = findSignificantContours(img, edgeImg_8u)

        # Mask
        mask = edgeImg.copy()
        mask[mask > 0] = 0
        cv2.fillPoly(mask, significant, 255)

        # Invert mask
        mask = np.logical_not(mask)

        #Finally remove the background
        img[mask] = 0

	#plt.imshow(img_show),plt.show()
	
	
	return img



