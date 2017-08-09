#! /usr/bin/env python
# -*- coding: utf-8 -*-
#CNNの特徴量をそのまま保存する.(PlaceCNN版)
#Akira Taniguchi 2017/01/22-2017/06/25-
import sys, os, os.path, caffe
import glob
import cv2
import numpy as np
#from sklearn.cluster import KMeans
from __init__ import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass

CNNfolder = '/home/akira/Dropbox/SpCoSLAM/PlaceCNN/'

if (Descriptor == "CNN_Place205"):
  #CNN_FILE = "placesCNN"
  #FULL PATH
  MEAN_FILE = CNNfolder + 'placesCNN/places205_mean.npy'
  #'/home/akira/Caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
  MODEL_FILE = CNNfolder + 'placesCNN/places205CNN_deploy.prototxt'
  #'/home/akira/Caffe/examples/imagenet/imagenet_feature.prototxt'
  PRETRAINED = CNNfolder + 'placesCNN/places205CNN_iter_300000.caffemodel'
  #'/home/akira/Caffe/examples/imagenet/caffe_reference_imagenet_model'
elif (Descriptor == "hybridCNN"):
  #CNN_FILE = "hybridCNN"
  #FULL PATH
  MEAN_FILE = CNNfolder + 'hybridCNN/hybridCNN__mean.npy'
  #'/home/akira/Caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
  MODEL_FILE = CNNfolder + 'hybridCNN/hybridCNN_deploy.prototxt'
  #'/home/akira/Caffe/examples/imagenet/imagenet_feature.prototxt'
  PRETRAINED = CNNfolder + 'hybridCNN/hybridCNN_iter_700000.caffemodel'
  #'/home/akira/Caffe/examples/imagenet/caffe_reference_imagenet_model'

LAYER = 'prob' #'fc6wi'
INDEX = 4

trialname = datasetfolder + dataset1
#"/home/akira/Dropbox/SpCoSLAM/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/"
filelist = glob.glob(trialname+"*.jpg")
filelist.sort()
Data = len(filelist)

#trialname = raw_input("trialname?(folder) >")
#start = raw_input("start number?>")
#end   = raw_input("end number?>")

#sn = int(start)
#en = int(end)
#Data = int(en) - int(sn) +1

#descriptors = []
#descriptors_bgr = []
#object_feature = [ [] for j in range(Data) ]
#object_color   = [ [] for j in range(Data) ]

#foldername = datafolder + trialname
#フォルダ作成
Makedir( trialname + Descriptor )

for imgname in filelist:
    #filename = foldername+str(trial+sn).zfill(3)+'/'

    #物体数の読み込み
    #gyo = 0
    #for line in open(filename + 'object_center.txt', 'r'):
    #  #itemList = line[:-1].split(',')
    #  if gyo == 0:
    #    object_num = int(line)
    #  gyo = gyo + 1    


    #for object in range(object_num):
    #imgname = filename + 'image/object_' + str(object+1).zfill(2) + '.ppm' 
    print imgname  
    
    timename = imgname[len(trialname):-4]

    net = caffe.Classifier(MODEL_FILE, PRETRAINED)
    #caffe.set_phase_test()
    caffe.set_mode_cpu()
    net.transformer.set_mean('data', np.load(MEAN_FILE))
    #net.set_mean
    #net.set_raw_scale
    net.transformer.set_raw_scale('data', 255)
    net.transformer.set_channel_swap('data', (2,1,0))
    image = caffe.io.load_image(imgname)
    net.predict([ image ])
    feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
    #print(','.join(map(str, feat)))
    fp = open(trialname+ Descriptor + "/" + timename+'.csv','w')
    fp.write(','.join(map(str, feat)))
    fp.close()
