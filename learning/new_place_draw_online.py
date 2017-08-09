#!/usr/bin/env python
# -*- coding:utf-8 -*-
#学習した場所領域のサンプルをrviz上に可視化するプログラム
#作成者 石伏智
#作成日 2015年12月
#サンプリング点プロット→ガウスの概形描画に変更（磯部、2016卒論）
#編集、更新：谷口彰 更新日：2017/02/10
#mu 2次元、sig 2×2次元版
#自己位置も取得して描画するのは別プログラム

"""
実行前に指定されているフォルダが正しいかをチェックする
file_read.pyも同様に ！

実行方法
python place_draw.py (parameterフォルダの絶対パス) (表示する場所領域を指定したい場合は数字を入力)

実行例
python place_draw.py /home/emlab/py-faster-rcnn/work/gibbs_sampling_program

"""

import glob
import re
import os
import rospy
import math
import sys
import time
import geometry_msgs.msg as gm
from geometry_msgs.msg import Point
import sensor_msgs.msg as sm
from  visualization_msgs.msg import Marker
from  visualization_msgs.msg import MarkerArray
import numpy as np
import struct
#import PyKDLs
sys.path.append("lib/")
from __init__ import *

"""
def read_result(filename):
  file_dir = os.chdir(filename)
  f = open('SBP.txt')
  line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
  place_num = int(line)
  return place_num
"""

def mu_read(filename):
    all_mu=[]
    #fp = open(filename+'mu'+maxparticle+".csv", "r") # check
    #convert = lambda text: int(text) if text.isdigit() else text
    #alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    #file.sort(key=alphanum_key)
    #for f in file:
    K = 0
    for line in open(filename+'mu'+str(maxparticle)+".csv", 'r'): #.readlines()
        mu=[] #(x,y,sin,cos)
        
        # readlines()は,ファイルを全て読み込み、1行毎に処理を行う
        #print line
        data=line[:].split(',')
        mu +=[float(data[0])]
        mu +=[float(data[1])]
        mu +=[0]#float(data[2])]
        mu +=[0]#float(data[3])]
        #print position
        all_mu.append(mu)
        K += 1
    return all_mu, K


def sigma_read(filename):
    all_sigma=[]
    #file = glob.glob(filename+'/parameter3/sigma/*.txt') # check
    #convert = lambda text: int(text) if text.isdigit() else text
    #alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    #file.sort(key=alphanum_key)
    for line in open(filename+'sig'+str(maxparticle)+".csv", 'r'):
        #sigma=[] #(x,y,sin,cos)
        data=line[:].split(',')
        sigma = [[float(data[0]),float(data[1]),0,0],[float(data[2]),float(data[3]),0,0],[0,0,0,0],[0,0,0,0]]
        # readlines()は,ファイルを全て読み込み、1行毎に処理を行う
        #line=open(f, 'r').readlines()
        #i = 0
        #for l in line:
        #    sigma_l.append(float(data[0]))
        #    sigma_l.append(float(data[1]))
        #    sigma_l.append(float(data[2]))
        #    sigma_l.append(float(data[3]))
        #    
        #sigma.append(sigma_l)
        #    
        all_sigma.append(sigma)
    return all_sigma

"""
def sampling_read(filename, class_num):
    c_all_position=[]
    for c in range(class_num):
        all_position=[] #すべての自己位置データのリスト
        
        file = glob.glob(filename+'/sampling_data3/class'+repr(c)+'/*.txt') # check
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
"""

# 自作ファイル
#import file_read as f_r
#from SBP import read_result

#実験ファイル名trialnameを取得
trialname = sys.argv[1]
print trialname

#step番号を取得
step = int(sys.argv[2])
print step

filename = datafolder+trialname+"/"+ str(step) +"/"
#filename50 = datafolder+trialname+"/"+ str(50) +"/"

#maxparticle = 0
#i = 0
##datafolder+trialname+"/"+stepにおける最大尤度のパーティクルを読み込み
#for line in open( filename50 + 'weights.csv', 'r'):
#      #itemList = line[:].split(',')
#      if (i == 0):
#        maxparticle = int(line)
#        i +=1

maxparticle = int(sys.argv[3]) #どのIDのパーティクルか
#pid  = int(sys.argv[3])


#filename=sys.argv[1]

#Class_NUM=0#read_result(filename)
RAD_90=math.radians(90)
color_all=1   #1 or 0 、(0ならばすべて赤)
mu_draw =1    #1 or 0 、(0ならば中心値を表示しない)
sigma_draw=1  #1 or 0, (0ならば分散を表示しない)
mu_arrow=0    #矢印を可視化する場合
COLOR=[
#[0,0,0], #ロボット自己位置用
[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0.5,0,0.5], #4
[0,0.5,0.5],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.6,0.2,0.2],#9
[0.2,0.6,0.2],[0.2,0.2,0.6],[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4], #14
[0.7,0.2,0.1],[0.7,0.1,0.2],[0.2,0.7,0.1],[0.1,0.7,0.2],[0.2,0.1,0.7],#19
[0.1,0.2,0.7],[0.5,0.2,0.3],[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.5,0.3],#24
[0.3,0.2,0.5],[0.2,0.3,0.5],[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7],#29
[0.6,0.3,0.1],[0.6,0.1,0.3],[0.1,0.6,0.3],[0.3,0.6,0.1],[0.3,0.1,0.6],#34
[0.1,0.3,0.6],[0.8,0.2,0],[0.8,0,0.2],[0.2,0.8,0],[0,0.8,0.2],#39
[0.2,0,0.8],[0,0.2,0.8],[0.7,0.3,0],[0.7,0,0.3],[0.3,0.7,0.0],#44
[0.3,0,0.7],[0,0.7,0.3],[0,0.3,0.7],[0.25,0.25,0.5],[0.25,0.5,0.25], #49
[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0.5,0,0.5], #54
[0,0.5,0.5],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.6,0.2,0.2],#59
[0.2,0.6,0.2],[0.2,0.2,0.6],[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4], #64
[0,7,0.2,0.1],[0.7,0.1,0.2],[0.2,0.7,0.1],[0.1,0.7,0.2],[0.2,0.1,0.7],#69
[0.1,0.2,0.7],[0.5,0.2,0.3],[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.5,0.3],#74
[0.3,0.2,0.5],[0.2,0.3,0.5],[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7],#79
[0.6,0.3,0.1],[0.6,0.1,0.3],[0.1,0.6,0.3],[0.3,0.6,0.1],[0.3,0.1,0.6],#84
[0.1,0.3,0.6],[0.8,0.2,0],[0.8,0,0.2],[0.2,0.8,0],[0,0.8,0.2],#89
[0.2,0,0.8],[0,0.2,0.8],[0.7,0.3,0],[0.7,0,0.3],[0.3,0.7,0.0],#94
[0.3,0,0.7],[0,0.7,0.3],[0,0.3,0.7],[0.25,0.25,0.5],[0.25,0.5,0.25] #99
]

#特定の番号のガウス分布のみ描画したいとき
try: 
    Number=None #int(sys.argv[3])
except IndexError:
    Number=None

# 石伏さんはハイパーパラメータの値をパラメータ.txtに保持しているため、以下の処理をしている
#env_para=np.genfromtxt(filename+"/パラメータ.txt",dtype= None,delimiter =" ")
#Class_NUM=int(env_para[4][1])

"""
#=============各場所領域に割り当てられているデータの読みこみ===================
def class_check():
    Class_list=[]
    for i in range(Class_NUM):
        #f=filename+"/parameter3/class/class"+repr(i)+".txt" # check
        data=[]
        # default(エラー)
        #for line in open(f,'r').readlines():
        #    print str(line) + "\n\n"
        #    data.append(int(line))
        
        #for line in open(f, 'r'):
        #    print "読み込み完了"
        
        #replaceを使えば簡単にできる
        #line1=line.split('[') # 始めの"["を除く
        #line1=line1[1].split(']') # 終わりの"["を除く
        #line2=line1[0]
        #print "\nline2:" + str(line2) + "\n"
        
        # 場所クラスに中身があるときはtry、中身がないときはexceptに移動
        #try:
        #    data = [int(item) for item in line2.split(',')]
        #except ValueError:
        #    data = []

        #c=[]

        #for item in data:
        #    print item
        #    try:
        #        num=int(item)
        #        c.append(num)
        #    except ValueError:
        #        pass
        Class_list.append(data)
    return Class_list
"""

def place_draw():
    # 場所のクラスの割り当てられていない場合は省く→CRPでは割り当てられていないデータは存在しない
    #class_list=class_check()
    #print class_list
    
    
    pub = rospy.Publisher('draw_space',MarkerArray, queue_size = 10)
    rospy.init_node('draw_spatial_concepts', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    #ロボットの自己位置を読み込み
    #mu_temp = [[float(sys.argv[4]),float(sys.argv[5]),0,0]]
    #sigma_temp = [[[0.1,0,0,0],[0,0.1,0,0],[0,0,0,0],[0,0,0,0]]]
    
    #最大尤度のパーティクルのmuとsigを読み込み
    mumu,Class_NUM = mu_read(filename)
    sigsig = sigma_read(filename)
    #sample = sampling_read(filename, Class_NUM)
    #print "sigma: ",sigma
    print sigsig
    #mu_all = mu_temp + mumu
    #sigma = sigma_temp + sigsig
    mu_all = mumu
    sigma = sigsig
    print mu_all
    print sigma
    #Class_NUM += 1
    
    data_class=[i for i in xrange(Class_NUM)]
    #for n in range(Class_NUM):
    #    #if len(class_list[n])!=0:
    #    data_class.append(n)
    
    marker_array=MarkerArray()
    id=0
    for c in data_class:
        #場所領域の中心値を示す場合
        #===場所領域の範囲の可視化====================
        if sigma_draw==1:
            
            marker =Marker()
            marker.type=Marker.CYLINDER
            
            (eigValues,eigVectors) = np.linalg.eig(sigma[c])
            angle = (math.atan2(eigVectors[1, 0], eigVectors[0, 0]));
            
            marker.scale.x = 2*math.sqrt(eigValues[0]);
            marker.scale.y = 2*math.sqrt(eigValues[1]);
            
            marker.pose.orientation.w = math.cos(angle*0.5);
            marker.pose.orientation.z = math.sin(angle*0.5);
            
            
            marker.scale.z=0.01 # default: 0.05
            marker.color.a=0.3
            marker.header.frame_id='map'
            marker.header.stamp=rospy.get_rostime()
            marker.id=id
            id +=1
            marker.action=Marker.ADD
            marker.pose.position.x=mu_all[c][0]
            marker.pose.position.y=mu_all[c][1]
            marker.color.r = COLOR[c][0] # default: COLOR[c][0] 色のばらつきを広げる
            marker.color.g = COLOR[c][1] # default: COLOR[c][1] 色のばらつきを広げる
            marker.color.b = COLOR[c][2] # default: COLOR[c][2] 色のばらつきを広げる

            if Number != None:
                if Number==c:
                    marker_array.markers.append(marker)
            else:
                    marker_array.markers.append(marker)
        if mu_draw==1:
            mu_marker =Marker()
            
            if mu_arrow==1: #矢印を可視化する場合
                mu_marker.type=Marker.ARROW
                orient_cos=mu_all[c][3]
                orient_sin=mu_all[c][2]
                if orient_sin>1.0:
                    orient_sin=1.0
                elif orient_sin<-1.0:
                    orient_sin=-1.0
                #radian xを導出
                radian=math.asin(orient_sin)
                if orient_sin>0 and orient_cos<0:
                    radian=radian+RAD_90
                elif orient_sin<0 and orient_cos<0:
                    radian=radian-RAD_90
            
                mu_marker.pose.orientation.z=math.sin(radian/2.0)
                mu_marker.pose.orientation.w=math.cos(radian/2.0)
                #<<<<<<<矢印の大きさ変更>>>>>>>>>>>>>>>>>>>>>>>>
                mu_marker.scale.x=0.5 # default: 0.4
                mu_marker.scale.y=0.07 # default: 0.1
                mu_marker.scale.z=0.001 # default: 1.0
                mu_marker.color.a=1.0
                
            elif mu_arrow==0:
                mu_marker.type=Marker.SPHERE
                mu_marker.scale.x=0.1
                mu_marker.scale.y=0.1
                mu_marker.scale.z=0.01 # default: 0.05
                mu_marker.color.a=1.0
            
            mu_marker.header.frame_id='map'
            mu_marker.header.stamp=rospy.get_rostime()
            mu_marker.id=id
            id +=1  
            mu_marker.action=Marker.ADD
            mu_marker.pose.position.x=mu_all[c][0]
            mu_marker.pose.position.y=mu_all[c][1]
            #print c,mu_marker.pose.position.x,mu_marker.pose.position.y
            
            if color_all==1:
                mu_marker.color.r = COLOR[c][0] # default: COLOR[c][0]
                mu_marker.color.g = COLOR[c][1] # default: COLOR[c][1]
                mu_marker.color.b = COLOR[c][2] # default: COLOR[c][2]
            elif color_all==0:
                mu_marker.color.r = 1.0
                mu_marker.color.g = 0
                mu_marker.color.b = 0
                
            if Number != None:
                if Number==c:
                    marker_array.markers.append(mu_marker)
            else:
                    marker_array.markers.append(mu_marker)

    print marker_array.markers
    count =0
    #while not rospy.is_shutdown():
    while(count <= 5):    
        #pub.publish(marker)
        pub.publish(marker_array)
        rate.sleep()
        #time.sleep(5.0)
        count = count+1
    

if __name__ == '__main__':
    try:
        place_draw()
    except rospy.ROSInterruptException:
        pass
