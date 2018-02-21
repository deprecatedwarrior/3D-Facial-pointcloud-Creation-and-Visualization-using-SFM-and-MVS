import numpy as np
import cv2
from scipy.signal import savgol_filter

#start with edge setection
#Uisng Sobel operator in the x and y directions and then using np.hypot for finding the sum of square of sides of a right triangle

def edgeDetect(channel):
	
	"""
	Performs edge detection on a given frame. If we're working on 3 channel images, this is to be done separately on every channel
	It is advised to perform Gaussian filtering prior to this to remove some noise before edge detection
	"""
	sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
	sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
	sobel = np.hypot(sobelx, sobely)
	# Some values seem to go above 255. However RGB channels has to be within 0-255
	sobel[sobel > 255] = 255;
	return sobel

def findSignificantContours (img, edgeImg):

	"""
	From the original image and the edgeImage, we perform contour detection
	Here, one contour can be a part of another contour hence we perform heirarchical contour detection
	If contour c1 is a part of contour c2, then c1 is a 'child' of c2
	OpenCV treats tree as a flat array with each tuple containing the index to the parent array
	"""

    	image, contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    	# Find level 1 contours
    	level1 = []
    	for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        	if tupl[3] == -1:
            		tupl = np.insert(tupl, 0, [i])
            		level1.append(tupl)

    	# From among them, we remove the contours that do no take at least 5% of the image area. This reduces noise further
    	significant = []
    	tooSmall = edgeImg.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
	window_size = int(round(min(img.shape[0], img.shape[1]) * 0.05)) # Consider each window to be 5% of image dimensions
    	for tupl in level1:
        	contour = contours[tupl[0]];
        	area = cv2.contourArea(contour)
		
		x = savgol_filter(contour[:,0,0], window_size * 2 + 1, 3,0,1.0,-1,'nearest',0.0) #these parameters can be tweaked for better results
	        y = savgol_filter(contour[:,0,1], window_size * 2 + 1, 3,0,1.0,-1,'nearest',0.0)

        	approx = np.empty((x.size, 1, 2))
        	approx[:,0,0] = x
        	approx[:,0,1] = y
        	approx = approx.astype(int)
        	contour = approx
		
        	if area > tooSmall:
            		cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
            		significant.append([contour, area])

    	significant.sort(key=lambda x: x[1])
	return [x[0] for x in significant]
