import os
import sys
import glob
import cv2
import dlib
import numpy as np
import argparse
from video2frames import Video2Frames as vid2frames
from facial_landmark import facial_landmark
from sfm_structure import SFMGetter
from preprocess import Preprocess 


# specify paths and parameters
def parse_args():
    args = {}
    args['testvideo'] = sys.argv[2]
    args['modelfile'] = sys.argv[4]
    args['savepath'] = sys.argv[6]
    return args

"""
1. Reads a test video and saves frames in cwd/data
2. Select two random frames for sparse facial point cloud construction
3. Subtract background. Uses Preprocess class to perform frame preprocessing
4. Compute facial landmarks using DLib's ensemble of regression trees
5. Get 3d points from SFMGetter class. It uses SIFT detector, BFMatcher
   and using iPhone 6s camera parameters to perform SFM. More details 
   in sfm_structure.py
6. Write 3D points in a csv
"""
if __name__ == '__main__':
    args = parse_args()
    testvideo = args['testvideo']
    savepath = args['savepath']
    modelfile = args['modelfile']
    cwd = os.getcwd()
    sample_video = os.path.join(cwd, testvideo)
    savepath = os.path.join(cwd, savepath)
    modelfile = os.path.join(cwd, modelfile)

    # Get frames
    vid = vid2frames(sample_video)
    vid_frames = vid.getFrames()
    paths = os.listdir(cwd, 'data')
    frame1_path = paths[80] # random
    frame2_path = paths[86] # random

    # preprocess frames
    prp = Preprocess()
    frame1_prep = prp.subtract_background(frame1_path)
    frame2_prep = prp.subtract_background(frame2_path)

    # get facial landmarks 
    landmarks1 = facial_landmark(frame1_prep)
    landmarks2 = facial_landmark(frame2_prep)

    # get 3D points for two frames 
    sfm = SFMGetter()
    points_3d = sfm.get3d(frame1_path, frame2_path, landmarks1, landmarks2)
    sfm.writeCSV(points_3d, savepath)