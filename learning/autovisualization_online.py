#!/usr/bin/env python
# -*- coding:utf-8 -*-
#自己位置と場所概念(位置分布)の可視化用プログラム（実行不可）
#Akira Taniguchi 2017/02/28 - 2017/09/01
import sys
import os
import random
import string
import signal
import subprocess
import time
import rospy
from std_msgs.msg import String
from __init__ import *

endstep = 50

# Reading particle data (ID,x,y,theta,weight,previousID)
def ReadParticleData(m_count, trialname):
  p = []
  for line in open ( datafolder + trialname + "/particle/" + str(m_count) + ".csv" ):
    itemList = line[:-1].split(',')
    p.append( Particle( int(itemList[0]), float(itemList[1]), float(itemList[2]), float(itemList[3]), float(itemList[4]), int(itemList[5])) )
  return p

# パーティクルIDの対応付け処理(Ct,itの対応付けも)
def ParticleSearcher(trialname):
  m_count = 0  #m_countの数
  #m_countのindexは1から始まる
  while (os.path.exists( datafolder + trialname + "/particle/" + str(m_count+1) + ".csv" ) == True):
    m_count += 1
  
  if (m_count == 0):  #エラー処理
    print "m_count",m_count
    #flag = 0
    #fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
    #fp.write(str(flag))
    #fp.close()
    #exit()
  
  #教示された時刻のみのデータにする
  #steplist = m_count2step(trialname, m_count)
  #step = len(steplist)
  #print steplist
  
  #C[1:t-1],I[1:t-1]のパーティクルID(step-1時点)と現在のparticleIDの対応付け
  #CTtemp = [[] for r in xrange(R)]
  #ITtemp = [[] for r in xrange(R)]
  #for particle in xrange(R):
  #  CTtemp[particle],ITtemp[particle] = ReaditCtData(trialname, step, particle)
  
  p = [[] for c in xrange(m_count)]
  for c in xrange(m_count):
    p[c] = ReadParticleData(c+1, trialname)        #m_countのindexは1から始まる   
    ######非効率なので、前回のパーティクル情報を使う（未実装）
  
  p_trajectory = [ [0.0 for c in xrange(m_count)] for i in xrange(R) ]
  #CT = [ [0 for s in xrange(step-1)] for i in xrange(R) ]
  #IT = [ [0 for s in xrange(step-1)] for i in xrange(R) ]
  
  for i in xrange(R):
    c_count = m_count-1  #一番最後の配列から処理
    #print c_count,i
    p_trajectory[i][c_count] = p[c_count][i]
    for c in xrange(m_count-1):  #0～最後から2番目の配列まで
      preID = p[c_count][p_trajectory[i][c_count].id].pid
      p_trajectory[i][c_count-1] = p[c_count-1][preID]
      #if (step == 1):
      #  CT[i] = CTtemp[i]
      #  IT[i] = ITtemp[i]
      #elif (step == 2):
      #  CT[i] = [1]
      #  IT[i] = [1]
      #else:
      #if (steplist[-2][0] == c_count): #CTtemp,ITtempを現在のパーティクルID順にする
      #    #CT[i] = [ CTtemp[preID][s] for s in xrange(step-1)]
      #    #IT[i] = [ ITtemp[preID][s] for s in xrange(step-1)]
      #    #print i,preID
      #print i, c, c_count-1, preID
      c_count -= 1
  
  X_To = [ [[p_trajectory[i][c].x,p_trajectory[i][c].y] for c in xrange(m_count)] for i in xrange(R) ]
  #for i in xrange(R):
  #  #X_To[i] = [p_trajectory[i][steplist[s][0]-1] for s in xrange(step)]
  #  X_To[i] = [[p_trajectory[i][m].x,p_trajectory[i][m].y] for c in xrange(m_count)]
  
  return X_To #, step, m_count #, CT, IT



# Reading particle data (ID,x,y,theta,weight,previousID)
def ReadParticleData2(step, particle, trialname):
  p = []
  pid = []
  for line in open ( datafolder + trialname + "/"+ str(step) + "/particle" + str(particle) + ".csv" ):
    itemList = line[:-1].split(',')
    p.append( [float(itemList[2]), float(itemList[3])] )
    pid.append( int(itemList[1]) )
    #p.append( Particle( int(itemList[0]), float(itemList[1]), float(itemList[2]), float(itemList[3]), float(itemList[4]), int(itemList[5])) )
  return p,pid

#roscoreは起動されているとする（上位プログラムで実行される）
#rvizは起動されているとする（上位プログラムで実行される）

#trialnameの取得
trialname = sys.argv[1]

#出力ファイル名を要求
#trialname = raw_input("trialname?(folder) >") #"tamd2_sig_mswp_01" 

#s=0
#m_count と教示stepとの対応付けを読み込み
list= []  #[ [m_count, step], ... ]
csvname = datafolder + trialname + "/m_count2step.csv"

for line in open ( csvname , 'r'):
      itemList = line[:-1].split(',')
      #print itemList
      list.append( [int(itemList[0]), int(itemList[1])] )
      #s += 1

end_m_count = list[-1][0]
m_list = [list[i][0] for i in range(len(list))]

filename50 = datafolder+trialname+"/"+ str(50) +"/"
maxparticle = 0
i = 0
##datafolder+trialname+"/"+stepにおける最大尤度のパーティクルを読み込み
for line in open( filename50 + 'weights.csv', 'r'):
      #itemList = line[:].split(',')
      if (i == 0):
        maxparticle = int(line)
        i +=1

#maxparticle = int(sys.argv[3])
#最終の教示ステップでの最大尤度のパーティクルの軌跡を取得
particle,pid = ReadParticleData2(50, maxparticle, trialname)# [0 for i in range(50)] 
XT = ParticleSearcher(trialname)
XTMAX = XT[maxparticle]

s = 0#15 #1
#m_count (step)のイテレーション
for m in range(1,end_m_count+1):#5,16):#5,7):#
  
  ##run_mapviewer.shを実行（trailname と m_countを指定）
  #map = "./run_mapviewer.sh "+trialname+" "+str(m)
  #map = "rosrun map_server map_server /home/akira/Dropbox/SpCoSLAM/data/"+ trialname+"/map/map"+str(m)+".yaml"
  #p = subprocess.Popen(map, shell=True)
  print list[s][0],m,s+1
  ##if (現在のm_countのstep == step):
  #if (m in m_list): #list[s][0] == m):
  ##########ここを実装すればよい↓##########
  ##オプション（trailname m_count particleのID ロボットのx座標 y座標）
  drawposition = "python ./new_position_draw_online.py "+trialname+" "+str(m)+" "+str(maxparticle)+" "+str(XTMAX[m-1][0])+" "+str(XTMAX[m-1][1])
  print drawposition
  p3 = subprocess.Popen(drawposition, shell=True)
  
  for s in range(len(list)):
   if(list[s][0] == m):
    ###new_place_draw_online.pyを実行（trialname 教示回数 particleのID）
    drawplace = "python ./new_place_draw_online.py "+trialname+" "+str(list[s][1])+" "+str(pid[s]) #+" "+str(particle[s][0])+" "+str(particle[s][1])
    print drawplace
    p2 = subprocess.Popen(drawplace, shell=True)
  ##########ここを実装すればよい↑##########
    #s = s+1
  time.sleep(2.0)


##rvizの画面を保存
#とりあえず画面そのものをキャプチャする
#git clone https://github.com/AtsushiSakai/jsk_visualization_packages.git
