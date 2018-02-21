# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
from numpy import newaxis
import argparse
import imutils
import dlib
import cv2

path_predictor = '/Users/avinashkaur/Desktop/WindowShopper/shape_predictor_68_face_landmarks.dat'
#path_img1 = '/Users/avinashkaur/Desktop/WindowShopper/data/frame80.jpg'
#path_img2 = '/Users/avinashkaur/Desktop/WindowShopper/data/frame86.jpg'

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

def landmarks_calculate_and_display(image, rects):
	# loop over the face detections
	predictor = dlib.shape_predictor(path_predictor)
	for (i, rect) in enumerate(rects):
        	# determine the facial landmarks for the face region, then
        	# convert the facial landmark (x, y)-coordinates to a NumPy
        	# array
		
		#converting input image to gray
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        	#shape = predictor(gray, rect) #reset to original for proper function
        	shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
        	#print shape

        	# convert dlib's rectangle to a OpenCV-style bounding box
        	# [i.e., (x, y, w, h)], then draw the face bounding box
        	(x, y, w, h) = face_utils.rect_to_bb(rect)
        	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        	# show the face number
        	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        	# loop over the (x, y)-coordinates for the facial landmarks
        	# and draw them on the image
        	for (x, y) in shape:
                	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		# show the output image with the face detections + facial landmarks
		#cv2.imshow("Output", image)
		#cv2.waitKey(0)
		return shape

def facial_landmark(img1, img2):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(path_predictor)

	# load the input image, (resize it), and convert it to grayscale
	#img1 = cv2.imread(path_img1)
	#img1 = imutils.resize(img1, width=500)

	#img2 = cv2.imread(path_img2)
	#img2 = imutils.resize(img2, width = 500)
	#gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #reset to original
	#gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #reset to original

	# detect faces in the grayscale image
	#rects1 = detector(gray1, 1) #reset to original for generic function
	#rects2 = detector(gray2, 1) #reset to original for generic function
	rects1 = detector(img1, 1) #for this to run in sfm script
	rects2 = detector(img2, 1) #for this to run in sfm script 
	shape1 = landmarks_calculate_and_display(img1, rects1)
	shape1 = shape1[:, newaxis, :]
	shape2 = landmarks_calculate_and_display(img2, rects2)
	shape2 = shape2[:, newaxis, :]
	return shape1, shape2

'''
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_predictor)

# load the input image, (resize it), and convert it to grayscale
img1 = cv2.imread(path_img1)
#img1 = imutils.resize(img1, width=500)

img2 = cv2.imread(path_img2)
#img2 = imutils.resize(img2, width = 500)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects1 = detector(gray1, 1)
rects2 = detector(gray2, 1)

shape1 = landmarks_calculate_and_display(img1, rects1)
shape1 = shape1[:, newaxis, :]
shape2 = landmarks_calculate_and_display(img2, rects2)
shape2 = shape2[:, newaxis, :]
'''

