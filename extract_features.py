#! /usr/bin/env python
# -*- coding: utf-8 -*-
# CNN feature extraction

import numpy as np
import sys, os, os.path, caffe, glob
import matplotlib.pyplot as plt

GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

MODELDIR = '../Caffe_model/'
DATADIR_ROOT = '../Data/'
SAVEDIR_ROOT = '../Data_feature/'

####################### Patemeters (below) ##############################################
DBNAME='Duke'#　database to apply 'VIPeR' or 'CUHK' or 'PRID' or 'GRID'
alpha = 0.5       # contrubution of combination attributes (0<= alpha <= 1), Higher value corresponds higher contributtion.
G = 7 	          # group number (fixed to 7)
r = 7 		  # combination number of attribute groups to combine  r \in { 2,3,4,5,6,7}
Nmin = 5          # threshold for the minimum image number of combination attributes.
subset = 1        # subset number in 7Cr attribute groups (it needs when r!=7)

ITES = ['50000']  #　iteration number of CNN fintuning to use feature extraction
		  #　ITES = ['5000', '10000', '50000', '100000']

LAYERS=['fc6']    #　layer of CNN features
		  #　LAYERS=['pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7']

####################### Patemeters (upto here) #####################################################

if DBNAME == 'VIPeR':
	DATADIR = DATADIR_ROOT + 'VIPeR/'
	CAMS=['cam_a/', 'cam_b/']
elif DBNAME == 'CUHK':
	DATADIR= DATADIR_ROOT +'CUHK01/'
elif DBNAME == 'PRID':
	DATADIR= DATADIR_ROOT +'prid_450s/'
	CAMS=['cam_a/', 'cam_b/']
elif DBNAME == 'GRID':
	DATADIR = DATADIR_ROOT + 'GRID/'
	CAMS=['gallery/', 'probe/']

elif DBNAME =='i-LID':
	DATADIR = DATADIR_ROOT + 'i-LIDS-VID/'
	CAMS=['images/cam1', 'images/cam2', 'sequences/cam1', 'sequences/cam2']

elif DBNAME == 'CUHK03':
	DATADIR = DATADIR_ROOT + 'CUHK03/'
	CAMS=['labeled/P1/cam1', 'labeled/P1/cam2', 'labeled/P2/cam1', 'labeled/P2/cam2',
	      'labeled/P3/cam1', 'labeled/P3/cam2',
	      'labeled/P4/cam1', 'labeled/P4/cam2', 'labeled/P5/cam1', 'labeled/P5/cam2',
	      'detected/P1/cam1', 'detected/P1/cam2', 'detected/P2/cam1', 'detected/P2/cam2',
	      'detected/P3/cam1', 'detected/P3/cam2',
	      'detected/P4/cam1', 'detected/P4/cam2', 'detected/P5/cam1', 'detected/P5/cam2']

elif DBNAME == 'Market':
	DATADIR = DATADIR_ROOT + 'Market-1501-v15.09.15/'
	CAMS=['bounding_box_test' , 'bounding_box_train', 'gt_bbox', 'gt_query', 'query']

elif DBNAME == 'Duke':
	DATADIR = DATADIR_ROOT + 'DukeMTMC-reID/'
	CAMS=['bounding_box_test' , 'bounding_box_train', 'query']

print len(CAMS)


# CNN name
exceptDB = DBNAME
EXPNAME =  "except_" + "_r"+str(r)+"Nmin"+str(Nmin)
if r != 7:
	EXPNAME = EXPNAME + "subset" + str(subset)
EXPNAME = EXPNAME + "+multi_"
C_EXPNAME = EXPNAME + 'alpha' + str(alpha)
EXP="FTAlex_" + str(G+1) + "Task"


for ITE in ITES:
	## load CNN
	MODEL_FILE = MODELDIR + EXP + '/Prototxt/Instance/Alexnet_' + C_EXPNAME + '_deploy.prototxt'
	PRETRAINED = MODELDIR + EXP + '/Snapshot/Alexnet_' + C_EXPNAME + '_train_iter_' + ITE + '.caffemodel'

	# Use ImageNet means for all pixels [104.00698793  116.66876762  122.67891434]
	meanval = np.array([104.0069873, 116.66876762, 122.67891434])
	net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=meanval, channel_swap=(2,1,0), raw_scale=255, image_dims=(227,227))

	INDEX = 4
	## CNN feature extraction and save
	if DBNAME == 'CUHK':
		SAVEDIR= SAVEDIR_ROOT + 'CUHK01/' + EXP + '/' + C_EXPNAME + 'ite' + ITE + '/'
	elif DBNAME == 'PRID':
		SAVEDIR= SAVEDIR_ROOT + 'prid_450s/' + EXP +  '/' + C_EXPNAME + 'ite' + ITE +'/'
	else:
		SAVEDIR= SAVEDIR_ROOT + DBNAME + '/'  + EXP + '/' + C_EXPNAME + 'ite' + ITE + '/'

	if DBNAME == 'Market' or DBNAME == 'Duke':
		# create save directory
		for c in range(len(CAMS)):
			for layer in LAYERS:
				dirname = SAVEDIR + layer + '/' + CAMS[c]
				if os.path.exists(dirname) == False:
					os.makedirs(dirname)

		for c in range(len(CAMS)):
			list = glob.glob( DATADIR+CAMS[c]+'/*.jpg')
			print DATADIR+CAMS[c]
			print list
			for file in list:
				print file
				image = caffe.io.load_image(file)
				#plt.imshow(image)

				file = file[len(DATADIR+CAMS[c]):len(file)]

				for layer in LAYERS:

					filesave = SAVEDIR + layer + '/' + CAMS[c] + file + '.txt'
					print filesave
					f = open(filesave, 'w')
					net.predict([ image ])
					feat = net.blobs[layer].data[INDEX].flatten().tolist()
					print len(feat)

					for d in range(len(feat)):
						f.writelines(str(feat[d])+" ")

					f.close()


	if DBNAME == 'CUHK03':
		# create save directory
		for c in range(20):
			for layer in LAYERS:
				dirname = SAVEDIR + layer + '/' + CAMS[c]
				if os.path.exists(dirname) == False:
					os.makedirs(dirname)

		for c in range(20):
			list = glob.glob( DATADIR+CAMS[c]+'/*.png')
			print DATADIR+CAMS[c]
			print list
			for file in list:
				print file
				image = caffe.io.load_image(file)
				#plt.imshow(image)

				file = file[len(DATADIR+CAMS[c]):len(file)]

				for layer in LAYERS:

					filesave = SAVEDIR + layer + '/' + CAMS[c] + file + '.txt'
					print filesave
					f = open(filesave, 'w')
					net.predict([ image ])
					feat = net.blobs[layer].data[INDEX].flatten().tolist()
					print len(feat)

					for d in range(len(feat)):
						f.writelines(str(feat[d])+" ")

					f.close()
