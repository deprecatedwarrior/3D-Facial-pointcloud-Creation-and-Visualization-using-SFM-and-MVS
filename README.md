# 3D-Facial-pointcloud-Creation-and-Visualization-using-SFM-and-MVS
This repository contains educational code for creating and visualizing a sparse facial point cloud for 3D reconstruction purposes.
The camera parameters have been customized for facial video capture on iPhone6S' front camera. 

Requirements in the Virtual environment
--------------------------------------------------
matplotlib==2.0.2
numpy==1.13.3
opencv-contrib-python==3.3.0.10
python-dateutil==2.6.1
scipy==1.0.0

File descriptions
--------------------------------------------------
sfm_structure.py 
--------------------------------------------------
- Reads two frames 
	(The frames are to be chosen manually for these programs, however, choice of frames is programmable; should be done after surface recontruction to optimize this process.)
- Finds interest points using SIFT detector and a Brute Force Matcher
	(From commercial standpoint, ORB to be used)
- Ratio test to store the top matches
- Apply DLIB's Facial Landmark Detector 
- Initialize SFM pipeline
- Camera parameters compatible to iPhone 6S. The requirement of the deliverable was for iPhone, hence the chosen paramters. 
- Estimate pose and convert to non homogeneous coordinates
- Export the 3D coordinates as a csv file 

--------------------------------------------------
background_sub.py
--------------------------------------------------
- Reads an image
- Gaussian blurring
- Edge detection
- Heirarchical Contour detection
- Background Subtraction

--------------------------------------------------
cannyedge.py
--------------------------------------------------
- Reads an image
- Uses OpenCV's Canny edge detection to output an edge map.

--------------------------------------------------
countframes.py
--------------------------------------------------
- helper python script to count frames
- embedded directly in other scripts
- given just for reference

--------------------------------------------------
facial_landmark.py
--------------------------------------------------
- Entire Facial Landmark detection pipeline
- takes trained model given as a .dat file 
- self-explanatory script


--------------------------------------------------
segmentation.py
--------------------------------------------------
- performs heirarchical contour detection based off of human faces
- imported in background_sub.py


--------------------------------------------------
sparse_cloud_using_vtk.py
--------------------------------------------------
-Uses Visualization Toolkit VTK Library for a good presentation of the point cloud. 

--------------------------------------------------
vid2frames.py
--------------------------------------------------
Takes a video as an input and outputs the frames

===================================================================================
Instructions for running the code:
1. Input video file to vid2frames script and give the directory for storing the output frames. 
2. Perform background subtraction on all the frames in the directory
3. In sfm_structure, choose the frames and get the output 3D points csv file
4ls
. On terminal, run python sparse_cloud_using_vtk.python test.csv 





