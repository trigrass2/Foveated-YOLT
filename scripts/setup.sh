#!/bin/bash

##### Functions

function get_network_weights
{
	#download pre-trained network model
        cd files

	# Caffenet
	wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
	
	# GoogLeNet
	wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
	
	# VGGNet 16 weight layers
	wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

	cd ..
}



# Get videos
get_network_weights


