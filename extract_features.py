import numpy as np
import sys, os, os.path, caffe, glob
import matplotlib.pyplot as plt

caffe.set_mode_gpu()

MODELDIR = 'Caffe_model/'               # directly of trained model
DATADIR_ROOT = '/mnt/HDD/Data/'         # root directly of data
SAVEDIR_ROOT = '/mnt/HDD/Data_feature/' # root directly to save extracted features

DBNAME='Market'  # database to extract CNN features
LAYERS=['fc6']  # layer name to extract CNN features
				# e.g., LAYERS=['pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7']

SAVEDIR= SAVEDIR_ROOT + DBNAME + '/FTAlex/'

## load CNN
if DBNAME == 'VIPeR' or DBNAME == 'CUHK' or DBNAME == 'PRID' or DBNAME == 'GRID':
	PRETRAINED = MODELDIR +  'Alexnet_PETA_except_' + DBNAME + '_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel'
else:
	PRETRAINED = MODELDIR +  'Alexnet_PETA_all_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel'

MODEL_FILE = MODELDIR + 'Alexnet_deploy.prototxt'

# Use ImageNet means for all pixels
meanval = np.array([104.0069873, 116.66876762, 122.67891434])
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=meanval, channel_swap=(2,1,0), raw_scale=255, image_dims=(227,227))

## setup of database
if DBNAME == 'VIPeR':
	DATADIR = DATADIR_ROOT + 'VIPeR/'
	CAMS=['cam_a/', 'cam_b/']
	EXT = '.bmp'

elif DBNAME == 'CUHK':
	DATADIR= DATADIR_ROOT +'CUHK01/'
	CAMS=['campus']
	EXT = '.png'

elif DBNAME == 'PRID':
	DATADIR= DATADIR_ROOT +'prid_450s/'
	CAMS=['cam_a/', 'cam_b/']
	EXT = '.png'

elif DBNAME == 'GRID':
	DATADIR = DATADIR_ROOT + 'GRID/'
	CAMS=['gallery/', 'probe/']
	EXT = '.jpeg'

elif DBNAME == 'Market':
	DATADIR = DATADIR_ROOT + 'Market-1501-v15.09.15/'
	CAMS=['bounding_box_test' , 'bounding_box_train', 'gt_bbox', 'gt_query', 'query']
	EXT = '.jpg'

elif DBNAME == 'Duke':
	DATADIR = DATADIR_ROOT + 'DukeMTMC-reID/'
	CAMS=['bounding_box_test' , 'bounding_box_train', 'query']
	EXT = '.jpg'

###################  CNN feature extraction and save  ########################################
# create save directory
for c in range(len(CAMS)):
	for layer in LAYERS:
		dirname = SAVEDIR + layer + '/' + CAMS[c]
		if os.path.exists(dirname) == False:
			os.makedirs(dirname)

# extract features and save
for c in range(len(CAMS)):
	list = glob.glob( DATADIR+CAMS[c]+'/*'+EXT)
	print DATADIR+CAMS[c]
	print list
	for file in list:
		print file
		image = caffe.io.load_image(file)
		file = file[len(DATADIR+CAMS[c]):len(file)]

		for layer in LAYERS:
			filesave = SAVEDIR + layer + '/' + CAMS[c] + file + '.csv'
			print filesave
			net.predict([ image ])
			feat = net.blobs[layer].data[4].flatten()
			print feat.size
			np.savetxt(filesave, feat, delimiter=',')
