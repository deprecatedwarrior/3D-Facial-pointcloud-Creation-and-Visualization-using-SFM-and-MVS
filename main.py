import glob
import cv2
import numpy as np
import os
from imutils import face_utils
from numpy import newaxis
import imutils
import dlib
from background_sub import preprocessing
from vid2frames import vid2frame
from facial_landmark import facial_landmark
#path = '/Users/avinashkaur/Desktop/WindowShopper/data/'

#Convert video to frames
#vid2frame()

#Set image paths
path_predictor = '/Users/avinashkaur/Desktop/WindowShopper/shape_predictor_68_face_landmarks.dat'
path_img1 = '/Users/avinashkaur/Desktop/WindowShopper/data/frame80.jpg'
path_img2 = '/Users/avinashkaur/Desktop/WindowShopper/data/frame86.jpg'

#Get facial landmarks
img1 = cv2.imread(path_img1)
img2 = cv2.imread(path_img2)
shape1, shape2 = facial_landmark(img1, img2)




#Get best match points from sfm
'''
try:
    if not os.path.exists('prep_data'):
        os.makedirs('prep_data')
except OSError:
    print ('Error: Creating directory of prep_data')

count = 0

for img in glob.glob('/Users/avinashkaur/Desktop/WindowShopper/data/*.jpg'): # All jpeg images
        proc_frame = preprocessing(img)
        name = './prep_data/proc_frame' + str(count) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, proc_frame)
        count +=1
'''



