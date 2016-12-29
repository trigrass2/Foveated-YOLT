#!/bin/bash

#if [ -z "$1" ] 
#  then
#    echo "No folder supplied!"
#    echo "Usage: bash `basename "$0"` mot_video_folder"
#    exit
#fi

DIR=$PWD;

# Choose which GPU the tracker runs on
GPU_ID=0

# Choose which video from the test set to start displaying
START_VIDEO_NUM=0

# Set to 0 to pause after each frame
PAUSE_VAL=1


# change this path to the absolute location of the network related files
FILES_FOLDER_ABSOLUTE_PATH=$PWD"/files/"
MODEL_FILE="deploy_caffenet.prototxt"
WEIGHTS_FILE="bvlc_reference_caffenet.caffemodel"
MEAN_FILE="imagenet_mean.binaryproto"
LABELS_FILE="val.txt"

# /home/filipa/Documents/Foveated_YOLT/files/ deploy_caffenet.prototxt bvlc_caffenet.caffemodel imagenet_mean.binaryproto val.txt
build/yolt $FILES_FOLDER_ABSOLUTE_PATH $CAFFE_MODEL $MODEL_FILE $WEIGHTS_FILE $MEAN_FILE $LABELS_FILE $GPU_ID
