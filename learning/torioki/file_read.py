#!/usr/bin/env python
# -*- coding:utf-8 -*-


# license removed for brevity
import glob
import re
import os
import rospy
import math
#from std_msgs.msg import String
import geometry_msgs.msg as gm
from geometry_msgs.msg import Point
import sensor_msgs.msg as sm
from  visualization_msgs.msg import Marker
from  visualization_msgs.msg import MarkerArray
import numpy as np
import struct

def mu_read(diric):
    all_mu=[]
    file = glob.glob(diric+'/parameter3/mu/*.txt') # check
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    file.sort(key=alphanum_key)
    for f in file:
        mu=[] #(x,y,sin,cos)
        
        # readlines()は,ファイルを全て読み込み、1行毎に処理を行う
        line=open(f, 'r').readlines()
        #print line
        data=line[0][:].split(',')
        mu +=[float(data[0])]
        mu +=[float(data[1])]
        mu +=[float(data[2])]
        mu +=[float(data[3])]
        #print position
        all_mu.append(mu)
    return all_mu


def sigma_read(diric):
    all_sigma=[]
    file = glob.glob(diric+'/parameter3/sigma/*.txt') # check
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    file.sort(key=alphanum_key)
    for f in file:
        sigma=[] #(x,y,sin,cos)
        
        # readlines()は,ファイルを全て読み込み、1行毎に処理を行う
        line=open(f, 'r').readlines()
        i = 0
        for l in line:
            sigma_l = []
            #print line
            data=l[:-1].split(',')
            #print data
            
            sigma_l.append(float(data[0]))
            sigma_l.append(float(data[1]))
            sigma_l.append(float(data[2]))
            sigma_l.append(float(data[3]))
            
            sigma.append(sigma_l)
            
        all_sigma.append(sigma)
    return all_sigma


def sampling_read(diric, class_num):
    c_all_position=[]
    for c in range(class_num):
        all_position=[] #すべての自己位置データのリスト
        
        file = glob.glob(diric+'/sampling_data3/class'+repr(c)+'/*.txt') # check
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        file.sort(key=alphanum_key)
        #print file
        for f in file:
            position=[] #(x,y,sin,cos)
            
            line=open(f, 'r').readlines()
            #print line
            data=line[0][:].split(',')
            position +=[float(data[0])]
            position +=[float(data[1])]
            position +=[float(data[2])]
            position +=[float(data[3])]
            #print position
            all_position.append(position)
        c_all_position.append(all_position)
    return c_all_position
