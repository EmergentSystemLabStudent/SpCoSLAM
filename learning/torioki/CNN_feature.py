#! /usr/bin/env python
# -*- coding: utf-8 -*-
#CNNの特徴量4096次元をそのまま保存する.
#Akira Taniguchi 2016/06/21-2016/08/21-
import sys, os, os.path, caffe
import cv2
import numpy as np
#from sklearn.cluster import KMeans
from __init__ import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass

#FULL PATH
MEAN_FILE = '/home/akira/Caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = '/home/akira/Caffe/examples/imagenet/imagenet_feature.prototxt'
PRETRAINED = '/home/akira/Caffe/examples/imagenet/caffe_reference_imagenet_model'
LAYER = 'fc6wi'
INDEX = 4


trialname = raw_input("trialname?(folder) >")
start = raw_input("start number?>")
end   = raw_input("end number?>")

sn = int(start)
en = int(end)
Data = int(en) - int(sn) +1

descriptors = []
descriptors_bgr = []
object_feature = [ [] for j in range(Data) ]
object_color   = [ [] for j in range(Data) ]

foldername = datafolder + trialname
#フォルダ作成
#Makedir( foldername+"("+str(sn).zfill(3)+"-"+str(en).zfill(3)+")" )


for trial in range(Data):
  filename = foldername+str(trial+sn).zfill(3)+'/'

  #物体数の読み込み
  gyo = 0
  for line in open(filename + 'object_center.txt', 'r'):
    #itemList = line[:-1].split(',')
    if gyo == 0:
      object_num = int(line)
    gyo = gyo + 1    
    #    for i in xrange(len(itemList)):
    #      if itemList[i] != '':
    #        Ct = Ct + [int(itemList[i])]



  for object in range(object_num):
    imgname = filename + 'image/object_' + str(object+1).zfill(2) + '.ppm' 
    print imgname  
    #img = cv2.imread(imgname)
    
    ###SIFT###
    #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #
    #sift = cv2.xfeatures2d.SIFT_create( contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6-0.8)
    ##nfeatures = 0, nOctaveLayers = 3,
    ##kp = sift.detect(gray,None)
    #
    #kp, dsc = sift.detectAndCompute(gray,None)
    
    
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
    fp = open(filename+'image/object_CNN_fc6_'+str(object+1).zfill(2)+'.csv','w')
    fp.write(','.join(map(str, feat)))
    fp.close()
    """
    dsc = []
    i = 0
    for line in open(filename+'image/object_CNN_fc6_'+str(object+1).zfill(2)+'.csv','r'):
      itemList = line[:-1].split(',')
      #print len(itemList)
      if i == 0:
      	for j in range(len(itemList)):
          dsc = dsc + [np.float32(itemList[j])]
      i = i + 1
    
    print len(dsc)#,sum(dsc)
    print dsc
    object_feature[trial] = object_feature[trial] + [dsc]
    #for i in range(len(dsc)):
    descriptors = descriptors + [np.array(dsc)]

    #descriptors = descriptors + [dsc]
    """

    #descriptors.append(dsc)
    #print descriptors 
    #print 'this is an example of a single SIFT keypoint:\n* * *'
    #explain_keypoint(kp[0])
    #img = cv2.drawKeypoints(gray, kp, img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    #cv2.imshow("img", img)
    #imgname = filename + 'image/sift_0' + repr(object+1) + '.png'
    #cv2.imwrite(imgname,img)

"""
des = np.array(descriptors)
print len(des),des
print len(object_feature),object_feature

#print kp
print 'CNN descriptors are vectors of shape', des[0].shape
  

#k=10
print "k-means(CNN), K =",k_sift

bow = cv2.BOWKMeansTrainer(k_sift)
cluster = bow.cluster(des)
#add = bow.add(dsc)
#KMeans(n_clusters=k_sift, init='k-means++').fit(des).cluster_centers_

#print k,len(cluster)
#print bow
print cluster

print np.sum(des),np.sum(cluster)

fp = open(foldername+"("+str(sn).zfill(3)+"-"+str(en).zfill(3)+")"+'/CNN_kmeans.csv','w')
for i in range(len(cluster)):
  fp.write(repr(i)+',')
  for j in range(len(cluster[i])):
    fp.write(repr(cluster[i][j])+',')
  fp.write('\n')
fp.close()



for trial in range(Data):
  filename = foldername+str(trial+sn).zfill(3)+'/'

  #物体数の読み込み
  gyo = 0
  for line in open(filename + 'object_center.txt', 'r'):
    #itemList = line[:-1].split(',')
    if gyo == 0:
      object_num = int(line)
    gyo = gyo + 1    
    #    for i in xrange(len(itemList)):
    #      if itemList[i] != '':
    #        Ct = Ct + [int(itemList[i])]

  for object in range(object_num):
    fp  = open(filename+'image/object_CNN_BoF_'+str(object+1).zfill(2)+'.csv','w')
    
    #min = 0
    BoF  = [0 for i in range(k_sift)]
    
    for f in range(len(object_feature[trial][object])):
      min = 0  #saveing nearest class
      min_value = 0.0
      for i in range(len(cluster)):
        dist = 0
        for j in range(len(cluster[i])):
          tmp = cluster[i][j] - object_feature[trial][object][f][j]
          dist = dist + (tmp * tmp)
        if(i == 0):
          min_value = dist
        elif(min_value > dist):
          min_value = dist
          min = i
      BoF[min] = BoF[min] + 1
  
    for i in range(k_sift):
      fp.write(repr(BoF[i])+',')
    fp.write('\n')
  
    fp.close()
"""





