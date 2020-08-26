import os
import cv2
import random
import csv
import numpy as np
import imutils
import dlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import matrix_rank
from imutils import face_utils
from numpy import newaxis
from facial_landmark import facial_landmark

##################################
##### ADD THE PATH HERE ##########

class SFMGetter:
	def __init__(self, ):
		self.top_matches = 800 # number of points to be matched
		self.K=[[1229.0,0.0,360.0,0.0,640.0,1153.0,0,0,1]]

	def get3D(self, path1, path2, shape1, shape2):
		'''
		Params: 
		- path1, path2 = two frames to perform sfm
		- shape1, shape2 = facial landmarks from frame 1 and frame2
		Use:
		- Reads two frames and returns the 3D world coordinates using SFM. 
		- Utilizes SIFT feature detector, Brute Force Matcher, 
		'''
		self.path1 = path1
		self.path2 = path2
		
		# Reading the images
		img1 = cv2.imread(self.path1)
		img2 = cv2.imread(self.path2)

		# Initiate SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()

		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None)

		# Drawing keypoints on the test image and the match_image
		img_training=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		img_matching=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		# Display the key point descriptors - sanity check
		# For display purposes we resize the image 
		# img_matching_display=cv2.resize(img_matching, (1200, 800))
		# plt.imshow(img_matching_display)
		# plt.show()

		# Count the number of keypoint descriptors in test and match image
		keypoint_test, w1=des1.shape
		keypoint_match, w2=des2.shape

		# BFMatcher with default params
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1,des2, k=2)
		# print matches

	 	# Apply ratio test
		good, pts1, pts2 = [], [], []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)


		# Sorting on the basis of the least distance concept in ascending order
		topMatches=sorted(good,key=lambda x:x.distance)

		# Creating map for the best 'top_count' matches after the good match criterion according to Lowe is set.
		# cv2.drawMatchesKnn expects list of lists as matches.
		img_match = cv2.drawMatches(img1,kp1,img2,kp2,topMatches[:self.top_matches],None,flags=2)

		#For displaying
		# image_match_display = cv2.resize(img_match, (1200,800))
		# plt.imshow(image_match_display)
		# plt.show()

		# Choosing only the top choices
		pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		#print pts2

		#Taking Facial landmark points from DLIB's landmark detection
		# shape1, shape2 = facial_landmark(img1, img2)
		
		#merging the key points obtained with the landmark points
		pts1 = np.vstack((pts1, shape1))
		pts2 = np.vstack((pts2, shape2))

		# Creating the Intrinsic Matrix K
		K = np.array(self.K).reshape(3,3)

		# Undistorting points for considering camera distortions
		upts1 = cv2.undistortPoints(pts1, K, distCoeffs=None)
		upts2 = cv2.undistortPoints(pts2, K, distCoeffs=None)

		# Calculating Essential Matrix
		E, mask = cv2.findEssentialMat(pts1,pts2,1229.0,(360.0, 640.0),cv2.RANSAC,0.999,1.0)
		rank2= matrix_rank(E)
		#print rank2
		#print E

		K_normal=[[1.0,0.0,0.0,0.0,1.0,0.0,0,0,1]]
		K_normal=np.array(K).reshape(3,3)

		# Decomposing E to get R and T
		Points, R,T,mask = cv2.recoverPose(E, upts1, upts2, K_normal)
		#print R
		#print T

		# Making the Projection matrices: 1->left; 2->right
		proj_mat2 = np.hstack((R,T))

		# Projection matrix of first cam at origin
		proj_mat1 = np.array([ [1,0,0,0],
				[0,1,0,0],
				[0,0,1,0]])

		# Changing dimensions of upts for triangulation
		upts1_array = np.array(upts1).reshape(2,len(pts1))
		upts2_array = np.array(upts2).reshape(2,len(pts2))

		# Converting 4D homogeneous coordinates to image coordinates
		P_l = np.dot(K,  proj_mat1)
		P_r = np.dot(K,  proj_mat2)
		pts4d = cv2.triangulatePoints(P_l, P_r, upts1, upts2)
		point_4d_nonHom = pts4d / np.tile(pts4d[-1, :], (4, 1))
		point_3d = point_4d_nonHom[:3, :].T
		#print point_3d
		return point_3d

	def writeCSV(self, points, savepath):
		with open(savepath, 'wb') as f:
			wtr = csv.writer(f, delimiter= ',')
			wtr.writerows(points)

		#Exporting to txt
		#np.savetxt("points_3d.csv", point_3d, delimiter="," , usecols=np.arange(0,2))

 		# Display the 3D points
		#fig=plt.figure()
 		#ax=Axes3D(fig)
		#ax.scatter(points[:,0],points[:,1],points[:,2],c='r',marker='.')
		#plt.show()
	
