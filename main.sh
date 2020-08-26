#!/bin/bash

testvideo='test_movie.mp4'
modelfile='shape_predictor_68_face_landmarks.dat'
savepath='points_3d.csv'

python main.py \
    --testvideo $testvideo \
    --modelfile $modelfile \
    --savepath $savepath 
