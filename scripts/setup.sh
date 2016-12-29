#!/bin/bash

##### Functions

function get_network_weights
{
	#download caffenet pre-trained network model
        cd files
	wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
	cd ..
}



# Get videos
get_network_weights


