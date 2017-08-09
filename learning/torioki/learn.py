#coding:utf-8

##############################################
# iCub learning program 
# Basically, Gibbs sampling of GMMs and LDA
# Akira Taniguchi 2016/05/31-2016/6/15
##############################################

###そもそもどんなデータを受け取るのか
#別プログラムで、処理しておいて、読み込むだけにする（？）
#単語は、latticelmする場合は、相互推定のプログラムを使いまわす
#画像は、背景差分、画像切り出し、物体座標取得、特徴量抽出をする必要がある⇒物体座標はiCubに戻す必要がある
#センサーモーターデータは、とりあえず、そのまま使う⇒K-meansと正規化
#アクションデータは、すでに物体座標によって相対座標に直されたデータが入るものとする。

###　流れ　###
#単語データ、iCubのセンサーモーターデータを読み込む
#ギブスサンプリングする（本プログラムの主要部分）
#ガウスをpyplotで描画（？）して画像として保存する
#学習したパラメータをファイルに保存する

#numpy.random.multivariate_normal
#http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.multivariate_normal.html
#scipy.stats.multivariate_normal
#http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.multivariate_normal.html
#scipy.stats.invwishart
#http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.invwishart.html
#numpy.random.dirichlet
#http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.dirichlet.html
#scipy.stats.dirichlet
#http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.dirichlet.html
#numpy.random.multinomial
#http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.multinomial.html
#scipy.stats.rv_discrete
#http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.rv_discrete.html

#---遂行タスク---#
#latticelmおよび相互推定も可能な様にコードを書く
#pygameを消す⇒pyplotとかにする
#いらないものを消す

#イテレーション100回ごとぐらいで結果を保存するようにする
##ｚのサンプリングで多項分布の事後確率がすべて0になる場合への対処

##共分散Σのロバストサンプリング

#---作業終了タスク---#
#文章の格フレームのランダムサンプル関数
#ファイル出力関数
#ギブスサンプリング
#順番：ｚ、π、φ、F、Θ
#ギブスサンプリングの挙動確認
#datadump/initial/ から初期値データと画像を読み込む？⇒画像処理プログラム
#出力ファイルの形式確認
#データなしの場合のガウスのサンプリングのロバスト処理
#データ読み込み関数
#muの初期値をk-meansで出す
#変なデータの精査（006）
####action の位置情報の正規化
####Mu_aのロバストサンプリング

#---保留---#
#Fの全探索を早くする方法
#処理の高速化

#import glob
#import codecs
#import re
import os
#import sys
import random
#import string
import collections
import itertools
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import multinomial,uniform,dirichlet
from scipy.stats import multivariate_normal,invwishart,rv_discrete
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
from sklearn.cluster import KMeans
from __init__ import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass

def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()

def Sample_Frame(num):
    modal = ["a","p","o","c"]
    other = "x"
    dis = 0
    F = []
    if (num >= len(modal)):
      #print "[Error] over modality"
      dis = num - len(modal)
      num = 4
      #return F
    for n in xrange(num):
      mo = int(uniform(0,4-n))
      F = F + [ modal[mo] ]
      modal.pop(mo)
    for i in xrange(dis):
      F.insert( int(uniform( 0, int(num+i+1) )), other )
    return F

def data_read(trialname,finename,sn,en):
    foldername = datafolder + trialname
    
    M = [0 for d in xrange(D)]
    N = [0 for d in xrange(D)]
    Ad = [0 for d in xrange(D)]
    
    w_dn = [ [] for d in xrange(D) ]
    a_d  = [ [] for d in xrange(D) ]
    o_dm = [ [] for d in xrange(D) ]
    c_dm = [ [] for d in xrange(D) ]
    p_dm = [ [] for d in xrange(D) ]
    
    #c = 0
    min_a = [10,10,10]   #仮の初期値
    max_a = [-10,-10,-10]  #仮の初期値
    min_o = 10000
    max_o = -10000
    
    for d in xrange(D):
      gyo = 0
      ##物体数M[d]の読み込み
      for line in open(foldername+str(d+sn).zfill(3)+'/object_center.txt','r'):
        if gyo == 0:
          M[d] = int(line)
          o_dm[d] = [ [ 0 for k in xrange(k_sift) ] for m in xrange(M[d]) ]
          c_dm[d] = [ [ 0 for k in xrange(k_rgb)  ] for m in xrange(M[d]) ]
          p_dm[d] = [ [ 0 for k in xrange(dim_p)  ] for m in xrange(M[d]) ]
          #elif gyo == 1:
          #  Ad[d] = int(line)-1 #物体番号1からMを0からM-1にする
        elif gyo == 1:
          dummy = 0
          #print "if random_obj was changed, but no problem."
        else:
          itemList = line[:-1].split(',')
          #for i in xrange(len(itemList)):
          if itemList[0] != '':
            #物体座標p_dmの読み込み
            p_dm[d][int(itemList[0])-1] = [float(itemList[1]),float(itemList[2])]#,float(itemList[3])]
        gyo = gyo + 1
      
      
      gyo = 0
      #対象物体の座標、手先座標読み込み
      for line in open(foldername+str(d+sn).zfill(3)+'/target_object.txt','r'):
        if gyo == 0:
          Ad[d] = int(line)-1
        elif gyo == 1:
          itemList = line[:-1].split(',')
          obj_pos = np.array([float(itemList[0]),float(itemList[1]),float(itemList[2])])
        elif gyo == 2:
          itemList = line[:-1].split(',')
          enf_pos = np.array([float(itemList[0]),float(itemList[1]),float(itemList[2])])
        elif gyo == 4:
          randomove = float(line)
        gyo = gyo + 1
        
      tmp = enf_pos - obj_pos #obj_pos - enf_pos
      if (d == 0):
         min_a = list(tmp)
         max_a = list(tmp)
      for i in xrange(3):
        if (min_a[i] > tmp[i]):
           min_a[i]= tmp[i]
        if (max_a[i] < tmp[i]):
           max_a[i] = tmp[i]
      a_d[d] = list(tmp) + [randomove] #相対3次元位置、指の曲げ具合
      
      gyo = 0
      for line in open(foldername+str(d+sn).zfill(3)+'/action.csv','r'):
        itemList = line[:-1].split(',')
        
        if ("" in itemList):
          itemList.pop(itemList.index(""))
        #関節箇所ごとに最小値と最大値で正規化
        if(gyo == 1 or gyo == 7 or gyo == 10):
          for i in xrange(len(itemList)):
            if gyo == 1: #head
              min = [-40,-70,-55,-35,-50, 0]
              max = [ 30, 60, 55, 15, 50,90]
            elif gyo == 7: #right_arm
              min = [-95,    0,-37,15.5,-90,-90,-20,  0,9.6,  0,  0,  0,  0,  0,   0,  0]
              max = [ 10,160.8, 80, 106, 90,  0, 40, 60, 90, 90,180, 90, 180, 90,180,270]
            elif gyo == 10: #torso
              min = [-50,-30,-10]
              max = [ 50, 30, 70]
            if itemList[i] != '':
              #print d,itemList[i]
              a_d[d] = a_d[d] + [ (float(itemList[i])-min[i])/float(max[i]-min[i]) ]  #正規化して配列に加える	
        if(gyo == 13):
          tactile = [0.0 for i in xrange(len(itemList)/12)]
          for i in xrange(len(itemList)/12):
            #for j in xrange(12):
            #  print i*12+j
            tactile[i] = sum([float(itemList[i*12+j]) for j in xrange(12)])/float(255*12)  #正規化して12個の平均
        
        gyo = gyo + 1
      a_d[d] = a_d[d] + tactile
      
      
      for m in xrange(M[d]):
        ##物体情報 BoF(SIFT)orCNN特徴の読み込み
        for line in open(foldername+str(d+sn).zfill(3)+'/image/object_' + Descriptor + '_'+str(m+1).zfill(2)+'.csv', 'r'):
          itemList = line[:-1].split(',')
          #print c
          #W_index = W_index + [itemList]
          for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              o_dm[d][m][i] = float(itemList[i])
              if min_o > o_dm[d][m][i]:
                min_o = o_dm[d][m][i]
              if max_o < o_dm[d][m][i]:
                max_o = o_dm[d][m][i]
              
        if (CNNmode == 0) or (CNNmode == -1):
          #BoFのカウント数を正規化
          sum_o_dm = sum(o_dm[d][m])
          for i in xrange(len(o_dm[d][m])):
            o_dm[d][m][i] = o_dm[d][m][i] / sum_o_dm
              
        ##色情報 BoF(RGB)の読み込み
        for line in open(foldername+str(d+sn).zfill(3)+'/image/object_RGB_BoF_'+str(m+1).zfill(2)+'.csv', 'r'):
          itemList = line[:-1].split(',')
          #print c
          #W_index = W_index + [itemList]
          for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              c_dm[d][m][i] = float(itemList[i])
        
        #BoFのカウント数を正規化
        sum_c_dm = sum(c_dm[d][m])
        for i in xrange(len(c_dm[d][m])):
          c_dm[d][m][i] = c_dm[d][m][i] / sum_c_dm
      
    #言語情報を取得
    d = 0
    for line in open(foldername +"("+str(sn).zfill(3)+"-"+str(en).zfill(3)+")"+'/' + trialname +'_words.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              w_dn[d] = w_dn[d] + [str(itemList[i])]
              N[d] = N[d] + 1
        d = d + 1
    
    print w_dn
    
    ##アクションデータの相対座標を正規化
    for d in xrange(D):
      a_d[d][0] = (a_d[d][0] - min_a[0]) / (max_a[0] - min_a[0])
      a_d[d][1] = (a_d[d][1] - min_a[1]) / (max_a[1] - min_a[1])
      a_d[d][2] = (a_d[d][2] - min_a[2]) / (max_a[2] - min_a[2])
      
      #CNN特徴の正規化
      if CNNmode == 1:
        for m in xrange(M[d]):
          for i in xrange(dim_o):
            o_dm[d][m][i] = (o_dm[d][m][i] - min_o) / (max_o - min_o)
    
    ###test data###
    #D = 10
    #M = [1 for d in xrange(D)]
    #N = [4 for d in xrange(D)]
    #Ad = [0 for d in xrange(D)]
    #basyo = [[],[-10+uniform(-1,1),-10+uniform(-1,1)], [10+uniform(-1,1),10+uniform(-1,1)], [0+uniform(-1,1),0+uniform(-1,1)]]
    #a_d  = [ basyo[1] , basyo[2] , basyo[3] , basyo[1] , basyo[2] , basyo[3] , basyo[1] , basyo[2] ]#, basyo[3] , basyo[1]]
    #p_dm = [[basyo[3]],[basyo[3]],[basyo[3]],[basyo[1]],[basyo[1]],[basyo[1]],[basyo[2]],[basyo[2]],[basyo[2]],[basyo[3]]]
    #o_dm = [[basyo[1]],[basyo[3]],[basyo[2]],[basyo[3]],[basyo[2]],[basyo[1]],[basyo[2]],[basyo[3]],[basyo[1]],[basyo[2]]]
    #c_dm = [[basyo[2]],[basyo[1]],[basyo[1]],[basyo[2]],[basyo[3]],[basyo[2]],[basyo[3]],[basyo[1]],[basyo[3]],[basyo[1]]]
    
    
    #w_dn = [["reach","front","box","green"] for i in range(1)]+[["touch","right","cup","green"] for i in range(1)]+[["touch","front","box","red"] for i in range(1)]+[["reach","front","box","blue"] for i in range(1)]+[["touch","front","box","green"] for i in range(1)]+[["lookat","front","cup","blue"] for i in range(1)]+[["lookat","left","cup","red"] for i in range(1)]+[["lookat","right","box","red"] for i in range(1)]#+[["a3a","p2p","o1o","c3c"] for i in range(1)]+[["a1a","p3p","o2o","c1c"] for i in range(1)]
    
    #読み込んだデータを保存
    fp = open( foldername +"("+str(sn).zfill(3)+"-"+str(en).zfill(3)+")"+'/' + filename +'/'+ trialname + '_' + filename +'_data.csv', 'w')
    for i in xrange(3):
      fp.write(repr(min_a[i])+',')
    fp.write('\n')
    for i in xrange(3):
      fp.write(repr(max_a[i])+',')
    fp.write('\n')
    fp.write('M\n')
    fp.write(repr(M))
    fp.write('\n')
    fp.write('N\n')
    fp.write(repr(N))
    fp.write('\n')
    fp.write('a_d\n')
    for d in xrange(D):
      fp.write(repr(a_d[d])+'\n')
    fp.write('\n')
    fp.write('p_dm\n')
    fp.write(repr(p_dm))
    fp.write('\n')
    fp.write('o_dm\n')
    for d in xrange(D):
      fp.write(repr(o_dm[d])+'\n')
    fp.write('\n')
    fp.write('c_dm\n')
    fp.write(repr(c_dm))
    fp.write('\n')
    fp.write('Ad\n')
    fp.write(repr(Ad))
    fp.write('\n')
    fp.write('w_dn\n')
    fp.write(repr(w_dn))
    fp.write('\n')
    fp.close()
    
    return M, N, w_dn, a_d, p_dm, o_dm, c_dm, Ad


def para_save(foldername,trialname,filename,za,zp,zo,zc,Fd,theta,W_list,Mu_a,Sig_a,Mu_p,Sig_p,Mu_o,Sig_o,Mu_c,Sig_c,pi_a,pi_p,pi_o,pi_c):
    foldername = foldername + "/" + filename
    trialname = trialname + "_" + filename
    #各パラメータ値を一つのファイルに出力
    fp = open( foldername +'/' + trialname +'_kekka.csv', 'w')
    fp.write('sampling_data\n') 
    fp.write('za\n')
    for d in xrange(D):
      fp.write(repr(d)+',')
    fp.write('\n')
    for d in xrange(D):
      fp.write(repr(za[d])+',')
    fp.write('\n')
    
    fp.write('zp\n')
    for d in xrange(D):
      for m in xrange(len(zp[d])):
        fp.write(repr(d)+'->'+repr(m)+',')
    fp.write('\n')
    for d in xrange(D):
      for m in xrange(len(zp[d])):
        fp.write(repr(zp[d][m])+',')
    fp.write('\n')
    
    fp.write('zo\n')
    for d in xrange(D):
      for m in xrange(len(zo[d])):
        fp.write(repr(d)+'->'+repr(m)+',')
    fp.write('\n')
    for d in xrange(D):
      for m in xrange(len(zo[d])):
        fp.write(repr(zo[d][m])+',')
    fp.write('\n')
    
    fp.write('zc\n')
    for d in xrange(D):
      for m in xrange(len(zc[d])):
        fp.write(repr(d)+'->'+repr(m)+',')
    fp.write('\n')
    for d in xrange(D):
      for m in xrange(len(zc[d])):
        fp.write(repr(zc[d][m])+',')
    fp.write('\n')
    
    fp.write('Fd\n')
    for d in xrange(D):
      fp.write(repr(d)+','+repr(Fd[d])+'\n')
      
    fp.write('theta\n,,')
    for w in xrange(len(W_list)):
      fp.write(repr(W_list[w])+',')
    fp.write('\n')
    for i in xrange(Ka):
      fp.write('a '+repr(i)+','+repr(i)+',')
      for w in xrange(len(W_list)):
        fp.write(repr(theta[i][w])+',')
      fp.write('\n')
    for i in xrange(Kp):
      fp.write('p '+repr(i)+','+repr(i+dict["p"])+',')
      for w in xrange(len(W_list)):
        fp.write(repr(theta[i+dict["p"]][w])+',')
      fp.write('\n')
    for i in xrange(Ko):
      fp.write('o '+repr(i)+','+repr(i+dict["o"])+',')
      for w in xrange(len(W_list)):
        fp.write(repr(theta[i+dict["o"]][w])+',')
      fp.write('\n')
    for i in xrange(Kc):
      fp.write('c '+repr(i)+','+repr(i+dict["c"])+',')
      for w in xrange(len(W_list)):
        fp.write(repr(theta[i+dict["c"]][w])+',')
      fp.write('\n')
    
    fp.write('action category\n')
    fp.write('Mu\n')
    for k in xrange(Ka):
      fp.write(repr(k)+',')
      for dim in xrange(dim_a):
        fp.write(repr(Mu_a[k][dim])+',')
      fp.write('\n')
    fp.write('Sig\n')
    for k in xrange(Ka):
      fp.write(repr(k)+'\n')
      for dim in xrange(dim_a):
        for dim2 in xrange(dim_a):
          fp.write(repr(Sig_a[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    
    fp.write('position category\n')
    fp.write('Mu\n')
    for k in xrange(Kp):
      fp.write(repr(k)+',')
      for dim in xrange(dim_p):
        fp.write(repr(Mu_p[k][dim])+',')
      fp.write('\n')
    fp.write('Sig\n')
    for k in xrange(Kp):
      fp.write(repr(k)+'\n')
      for dim in xrange(dim_p):
        for dim2 in xrange(dim_p):
          fp.write(repr(Sig_p[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    
    fp.write('object category\n')
    fp.write('Mu\n')
    for k in xrange(Ko):
      fp.write(repr(k)+',')
      for dim in xrange(dim_o):
        fp.write(repr(Mu_o[k][dim])+',')
      fp.write('\n')
    fp.write('Sig\n')
    for k in xrange(Ko):
      fp.write(repr(k)+'\n')
      for dim in xrange(dim_o):
        for dim2 in xrange(dim_o):
          fp.write(repr(Sig_o[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    
    fp.write('color category\n')
    fp.write('Mu\n')
    for k in xrange(Kc):
      fp.write(repr(k)+',')
      for dim in xrange(dim_c):
        fp.write(repr(Mu_c[k][dim])+',')
      fp.write('\n')
    fp.write('Sig\n')
    for k in xrange(Kc):
      fp.write(repr(k)+'\n')
      for dim in xrange(dim_c):
        for dim2 in xrange(dim_c):
          fp.write(repr(Sig_c[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    
    fp.write('pi_a'+',')
    for k in xrange(Ka):
      fp.write(repr(pi_a[k])+',')
    fp.write('\n')
    fp.write('pi_p'+',')
    for k in xrange(Kp):
      fp.write(repr(pi_p[k])+',')
    fp.write('\n')
    fp.write('pi_o'+',')
    for k in xrange(Ko):
      fp.write(repr(pi_o[k])+',')
    fp.write('\n')
    fp.write('pi_c'+',')
    for k in xrange(Kc):
      fp.write(repr(pi_c[k])+',')
    fp.write('\n')
    
    fp.close()
    
    #print 'File Output Successful!(filename:'+filename+')\n'
    
    ##パラメータそれぞれをそれぞれのファイルとしてはく
    fp = open(foldername +'/' + trialname +'_za.csv', 'w')
    for d in xrange(D):
      fp.write(repr(za[d])+',')
    fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_zp.csv', 'w')
    for d in xrange(D):
      for m in xrange(len(zp[d])):
        fp.write(repr(zp[d][m])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_zo.csv', 'w')
    for d in xrange(D):
      for m in xrange(len(zo[d])):
        fp.write(repr(zo[d][m])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_zc.csv', 'w')
    for d in xrange(D):
      for m in xrange(len(zc[d])):
        fp.write(repr(zc[d][m])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Fd.csv', 'w')
    #fp.write('Fd\n')
    for d in xrange(D):
      for f in xrange(len(Fd[d])):
        fp.write(repr(Fd[d][f])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_theta.csv', 'w')
    #fp.write('theta\n')
    for i in xrange(L):
      for w in xrange(len(W_list)):
        fp.write(repr(theta[i][w])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_W_list.csv', 'w')
    for w in xrange(len(W_list)):
      fp.write(repr(W_list[w])+',')
    fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Mu_a.csv', 'w')
    #fp.write('action category\n')
    #fp.write('Mu\n')
    for k in xrange(Ka):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_a):
        fp.write(repr(Mu_a[k][dim])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Sig_a.csv', 'w')
    #fp.write('Sig\n')
    for k in xrange(Ka):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_a):
        for dim2 in xrange(dim_a):
          fp.write(repr(Sig_a[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Mu_p.csv', 'w')
    #fp.write('position category\n')
    #fp.write('Mu\n')
    for k in xrange(Kp):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_p):
        fp.write(repr(Mu_p[k][dim])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Sig_p.csv', 'w')
    #fp.write('Sig\n')
    for k in xrange(Kp):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_p):
        for dim2 in xrange(dim_p):
          fp.write(repr(Sig_p[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Mu_o.csv', 'w')
    #fp.write('object category\n')
    #fp.write('Mu\n')
    for k in xrange(Ko):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_o):
        fp.write(repr(Mu_o[k][dim])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Sig_o.csv', 'w')
    #fp.write('Sig\n')
    for k in xrange(Ko):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_o):
        for dim2 in xrange(dim_o):
          fp.write(repr(Sig_o[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Mu_c.csv', 'w')
    #fp.write('color category\n')
    #fp.write('Mu\n')
    for k in xrange(Kc):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_c):
        fp.write(repr(Mu_c[k][dim])+',')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_Sig_c.csv', 'w')
    #fp.write('Sig\n')
    for k in xrange(Kc):
      #fp.write(repr(k)+',')
      for dim in xrange(dim_c):
        for dim2 in xrange(dim_c):
          fp.write(repr(Sig_c[k][dim][dim2])+',')
        fp.write('\n')
      fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_pi_a.csv', 'w')
    #fp.write('pi_a'+',')
    for k in xrange(Ka):
      fp.write(repr(pi_a[k])+',')
    fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_pi_p.csv', 'w')
    #fp.write('pi_p'+',')
    for k in xrange(Kp):
      fp.write(repr(pi_p[k])+',')
    fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_pi_o.csv', 'w')
    #fp.write('pi_o'+',')
    for k in xrange(Ko):
      fp.write(repr(pi_o[k])+',')
    fp.write('\n')
    fp.close()
    
    fp = open(foldername +'/' + trialname +'_pi_c.csv', 'w')
    #fp.write('pi_c'+',')
    for k in xrange(Kc):
      fp.write(repr(pi_c[k])+',')
    fp.write('\n')
    fp.close()
    
    print 'File Output Successful!(filename:'+foldername+')\n'
    
    

# Simulation
def simulate(foldername,trialname,filename,sn,en, M, N, w_dn, a_d, p_dm, o_dm, c_dm, Ad):
      np.random.seed()
      print w_dn
      ##各パラメータ初期化処理
      print u"Initialize Parameters..."
      za = [ int(uniform(0,Ko)) for d in xrange(D) ]
      zp = [ [ int(uniform(0,Kp)) for m in xrange(M[d]) ] for d in xrange(D) ]
      zo = [ [ int(uniform(0,Ko)) for m in xrange(M[d]) ] for d in xrange(D) ]
      zc = [ [ int(uniform(0,Kc)) for m in xrange(M[d]) ] for d in xrange(D) ]
      
      Fd = [ Sample_Frame(N[d]) for d in xrange(D)] #初期値はランダムに設定("a","p","o","c")
      
      cw = np.sum([collections.Counter(w_dn[d]) for d in xrange(D)])
      W_list = list(cw)  ##単語のリスト
      W = len(cw)  ##単語の種類数のカウント
      theta = [ sum(dirichlet(np.array([gamma for w in xrange(W)]),100))/100.0 for i in xrange(L) ] #indexと各モダリティーの対応付けはdictionary形式で呼び出す
      
      #KMeans(n_clusters=Ka, init='k-means++').fit(a_d).cluster_centers_
      Mu_a  = KMeans(n_clusters=Ka, init='k-means++').fit(a_d).cluster_centers_#[ np.array([uniform(mu_a_init[0],mu_a_init[1]) for i in xrange(dim_a)]) for k in xrange(Ka) ]
      Sig_a = [ np.eye(dim_a)*sig_a_init for k in xrange(Ka) ]
      #print "Mu_a",Mu_a
      
      p_temp = []
      for d in xrange(D):
        p_temp = p_temp + p_dm[d]
      Mu_p  = KMeans(n_clusters=Kp, init='k-means++').fit(p_temp).cluster_centers_#[ np.array([uniform(mu_p_init[0],mu_p_init[1]) for i in xrange(dim_p)]) for k in xrange(Kp) ]
      Sig_p = [ np.eye(dim_p)*sig_p_init for k in xrange(Kp) ]
      
      o_temp = []
      for d in xrange(D):
        o_temp = o_temp + o_dm[d]
      Mu_o  = KMeans(n_clusters=Ko, init='k-means++').fit(o_temp).cluster_centers_#[ np.array([uniform(mu_o_init[0],mu_o_init[1]) for i in xrange(dim_o)]) for k in xrange(Ko) ]
      Sig_o = [ np.eye(dim_o)*sig_o_init for k in xrange(Ko) ]
      
      c_temp = []
      for d in xrange(D):
        c_temp = c_temp + c_dm[d]
      Mu_c  = KMeans(n_clusters=Kc, init='k-means++').fit(c_temp).cluster_centers_#[ np.array([uniform(mu_c_init[0],mu_c_init[1]) for i in xrange(dim_c)]) for k in xrange(Kc) ]
      Sig_c = [ np.eye(dim_c)*sig_c_init for k in xrange(Kc) ]
      
      
      if nonpara == 0 :
        pi_a = sum(dirichlet([ alpha_a for c in xrange(Ka)],100))/100.0 #stick_breaking(gamma, L)#
        pi_p = sum(dirichlet([ alpha_p for c in xrange(Kp)],100))/100.0 #stick_breaking(gamma, L)#
        pi_o = sum(dirichlet([ alpha_o for c in xrange(Ko)],100))/100.0 #stick_breaking(gamma, L)#
        pi_c = sum(dirichlet([ alpha_c for c in xrange(Kc)],100))/100.0 #stick_breaking(gamma, L)#
      elif nonpara == 1:
        pi_a = stick_breaking(alpha_a, Ka) #sum(dirichlet([ alpha_a for c in xrange(Ka)],100))/100.0 #
        pi_p = stick_breaking(alpha_p, Kp) #sum(dirichlet([ alpha_p for c in xrange(Kp)],100))/100.0 #stick_breaking(gamma, L)#
        pi_o = stick_breaking(alpha_o, Ko) #sum(dirichlet([ alpha_o for c in xrange(Ko)],100))/100.0 #stick_breaking(gamma, L)#
        pi_c = stick_breaking(alpha_c, Kc) #sum(dirichlet([ alpha_c for c in xrange(Kc)],100))/100.0 #stick_breaking(gamma, L)#
      
      print theta
      print pi_a
      print pi_p
      print pi_o
      print pi_c
      print Mu_a
      print Mu_p
      print Mu_o
      print Mu_c
      print Fd
      
      ###初期値を保存(このやり方でないと値が変わってしまう)
      za_init = [ za[d] for d in xrange(D)]
      zp_init = [ [ zp[d][m] for m in xrange(M[d]) ] for d in xrange(D) ]
      zo_init = [ [ zo[d][m] for m in xrange(M[d]) ] for d in xrange(D) ]
      zc_init = [ [ zc[d][m] for m in xrange(M[d]) ] for d in xrange(D) ]
      Fd_init = [ Fd[d] for d in xrange(D)]
      theta_init = [ np.array(theta[i]) for i in xrange(L) ] 
      Mu_a_init  = [ np.array(Mu_a[k]) for k in xrange(Ka) ]
      Sig_a_init = [ np.array(Sig_a[k]) for k in xrange(Ka) ]
      Mu_p_init  = [ np.array(Mu_p[k]) for k in xrange(Kp) ]
      Sig_p_init = [ np.array(Sig_p[k]) for k in xrange(Kp) ]
      Mu_o_init  = [ np.array(Mu_o[k]) for k in xrange(Ko) ]
      Sig_o_init = [ np.array(Sig_o[k]) for k in xrange(Ko) ]
      Mu_c_init  = [ np.array(Mu_c[k]) for k in xrange(Kc) ]
      Sig_c_init = [ np.array(Sig_c[k]) for k in xrange(Kc) ]
      pi_a_init = [pi_a[k] for k in xrange(Ka)]
      pi_p_init = [pi_p[k] for k in xrange(Kp)]
      pi_o_init = [pi_o[k] for k in xrange(Ko)]
      pi_c_init = [pi_c[k] for k in xrange(Kc)]
      
      #初期パラメータのセーブ
      #filename_init = filename + "/init"
      trialname_init = "init/" + trialname 
      para_save(foldername,trialname_init,filename,za_init,zp_init,zo_init,zc_init,Fd_init,theta_init,W_list,Mu_a_init,Sig_a_init,Mu_p_init,Sig_p_init,Mu_o_init,Sig_o_init,Mu_c_init,Sig_c_init,pi_a_init,pi_p_init,pi_o_init,pi_c_init)
      
      ######################################################################
      ####                       ↓学習フェーズ↓                       ####
      ######################################################################
      print u"- <START> Learning of Lexicon and Multiple Categories ver. iCub MODEL. -"
      
      for iter in xrange(num_iter):   #イテレーションを行う
        print '----- Iter. '+repr(iter+1)+' -----'
        
        ########## ↓ ##### zaのサンプリング ##### ↓ ##########
        print u"Sampling za..."
        
        for d in xrange(D):         #データごとに
          temp = np.array(pi_a)
          for k in xrange(Ka):      #カテゴリ番号ごとに
            for n in xrange(N[d]):  #文中の単語ごとに
              if Fd[d][n] == "a":
                temp[k] = temp[k] * theta[k + dict["a"]][W_list.index(w_dn[d][n])]
            temp[k] = temp[k] * multivariate_normal.pdf(a_d[d], mean=Mu_a[k], cov=Sig_a[k])
          
          temp = temp / np.sum(temp)  #正規化
          za[d] = list(multinomial(1,temp)).index(1)
        print za
        ########## ↑ ##### zaのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### zpのサンプリング ##### ↓ ##########
        print u"Sampling zp..."
        
        for d in xrange(D):         #データごとに
          for m in xrange(M[d]):    #物体ごとに
            temp = np.array(pi_p)
            for k in xrange(Kp):      #カテゴリ番号ごとに
              for n in xrange(N[d]):  #文中の単語ごとに
                if Fd[d][n] == "p" and Ad[d] == m:
                  temp[k] = temp[k] * theta[k + dict["p"]][W_list.index(w_dn[d][n])]
              temp[k] = temp[k] * multivariate_normal.pdf(p_dm[d][m], mean=Mu_p[k], cov=Sig_p[k])
            
            temp = temp / np.sum(temp)  #正規化
            zp[d][m] = list(multinomial(1,temp)).index(1)
        print zp
        ########## ↑ ##### zpのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### zoのサンプリング ##### ↓ ##########
        print u"Sampling zo..."
        
        for d in xrange(D):         #データごとに
          for m in xrange(M[d]):    #物体ごとに
            temp = np.array(pi_o)
            logtemp = np.array([log(pi_o[k]) for k in xrange(Ko)])
            for k in xrange(Ko):      #カテゴリ番号ごとに
              for n in xrange(N[d]):  #文中の単語ごとに
                if Fd[d][n] == "o" and Ad[d] == m:
                  #temp[k] = temp[k] * theta[k + dict["o"]][W_list.index(w_dn[d][n])]
                  logtemp[k] = logtemp[k] + log(theta[k + dict["o"]][W_list.index(w_dn[d][n])])
              loggauss = multivariate_normal.logpdf(o_dm[d][m], mean=Mu_o[k], cov=Sig_o[k])
              #print loggauss
              logtemp[k] = logtemp[k] + loggauss#multivariate_normal.pdf(o_dm[d][m], mean=Mu_o[k], cov=Sig_o[k])
            
            logtemp = logtemp - np.max(logtemp)
            logtemp = logtemp - sp.misc.logsumexp(logtemp)#log(np.sum(np.array([exp(logtemp[k]) for k in xrange(Ko)])))  #正規化
            #print logtemp,sp.misc.logsumexp(logtemp)
            #print np.exp(logtemp),np.sum(np.exp(logtemp))
            zo[d][m] = list( multinomial(1,np.exp(logtemp)) ).index(1)
        print zo
        ########## ↑ ##### zoのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### zcのサンプリング ##### ↓ ##########
        print u"Sampling zc..."
        
        for d in xrange(D):         #データごとに
          for m in xrange(M[d]):    #物体ごとに
            temp = np.array(pi_c)
            for k in xrange(Kc):      #カテゴリ番号ごとに
              for n in xrange(N[d]):  #文中の単語ごとに
                if Fd[d][n] == "c" and Ad[d] == m:
                  #print temp
                  temp[k] = temp[k] * theta[k + dict["c"]][W_list.index(w_dn[d][n])]
                  #print temp
              temp[k] = temp[k] * multivariate_normal.pdf(c_dm[d][m], mean=Mu_c[k], cov=Sig_c[k])
              #print temp[k],multivariate_normal.pdf(c_dm[d][m], mean=Mu_c[k], cov=Sig_c[k])
            #print temp
            temp = temp / np.sum(temp)  #正規化
            #print temp
            zc[d][m] = list(multinomial(1,temp)).index(1)
        print zc
        ########## ↑ ##### zcのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### π_aのサンプリング ##### ↓ ##########
        print u"Sampling PI_a..."
        
        cc = collections.Counter(za)
        if nonpara == 0:
          temp = np.array([cc[k] + alpha_a for k in xrange(Ka)])
        elif nonpara == 1:
          temp = np.array([cc[k] + (alpha_a / float(Ka)) for k in xrange(Ka)])
        
        #加算したデータとパラメータから事後分布を計算しサンプリング
        pi_a = dirichlet(temp)
        print pi_a
        ########## ↑ ##### π_aのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### π_pのサンプリング ##### ↓ ##########
        print u"Sampling PI_p..."
        
        cc = np.sum([collections.Counter(zp[d]) for d in range(D)])
        if nonpara == 0:
          temp = np.array([cc[k] + alpha_p for k in xrange(Kp)])
        elif nonpara == 1:
          temp = np.array([cc[k] + (alpha_p / float(Kp)) for k in xrange(Ka)])
        
        #加算したデータとパラメータから事後分布を計算しサンプリング
        pi_p = dirichlet(temp)
        print pi_p
        ########## ↑ ##### π_pのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### π_oのサンプリング ##### ↓ ##########
        print u"Sampling PI_o..."
        
        cc = np.sum([collections.Counter(zo[d]) for d in range(D)])
        if nonpara == 0:
          temp = np.array([cc[k] + alpha_o for k in xrange(Ko)])
        elif nonpara == 1:
          temp = np.array([cc[k] + (alpha_o / float(Ko)) for k in xrange(Ka)]) 
        
        #加算したデータとパラメータから事後分布を計算しサンプリング
        pi_o = dirichlet(temp)
        print pi_o
        ########## ↑ ##### π_oのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### π_cのサンプリング ##### ↓ ##########
        print u"Sampling PI_c..."
        
        cc = np.sum([collections.Counter(zc[d]) for d in range(D)])
        if nonpara == 0:
          temp = np.array([cc[k] + alpha_c for k in xrange(Kc)])
        elif nonpara == 1:
          temp = np.array([cc[k] + (alpha_c / float(Kc)) for k in xrange(Ka)])
        
        #加算したデータとパラメータから事後分布を計算しサンプリング
        pi_c = dirichlet(temp)
        print pi_c
        ########## ↑ ##### π_cのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### μa,Σa(ガウス分布の平均、共分散行列)のサンプリング ##### ↓ ##########
        print u"Sampling myu_a,Sigma_a..."
        
        cc = collections.Counter(za)
        for k in xrange(Ka) : 
          nk = cc[k]
          xt = []
          m_ML = np.zeros(dim_a)
          ###kについて、zaが同じものを集める
          if nk != 0 :  #もしzaの中にkがあれば(計算短縮処理)        ##0ワリ回避
            for d in xrange(D) : 
              if za[d] == k : 
                xt = xt + [ np.array(a_d[d]) ]
            
            m_ML = sum(xt) / float(nk) #fsumではダメ
            #print "n:%d m_ML:%s" % (nk,str(m_ML))
            print "a%d n:%d" % (k,nk)
            
            ##ハイパーパラメータ更新
            kN = k0a + nk
            mN = ( k0a*m0a + nk*m_ML ) / kN  #dim_a 次元横ベクトル
            nN = n0a + nk
            VN = V0a + sum([np.dot(np.array([xt[j]-m_ML]).T,np.array([xt[j]-m_ML])) for j in xrange(nk)]) + (k0a*nk/kN)*np.dot(np.array([m_ML-m0a]).T,np.array([m_ML-m0a]))
            
            ##3.1##Σを逆ウィシャートからサンプリング
            Sig_a[k] = invwishart.rvs(df=nN, scale=VN) #/ n0a
            ##3.2##μを多変量ガウスからサンプリング
            Mu_a[k] = np.mean([multivariate_normal.rvs(mean=mN, cov=Sig_a[k]/kN) for i in xrange(100)],0) #サンプリングをロバストに
          else:  #データがないとき
            Mu_a[k]  = np.array([uniform(mu_a_init[0],mu_a_init[1]) for i in xrange(dim_a)])
            Sig_a[k] = invwishart.rvs(df=n0a, scale=V0a )#np.eye(dim_a)*sig_a_init
          
          
          
          if (nk != 0):  #データなしは表示しない
            print 'Mu_a '+str(k)+' : '+str(Mu_a[k])
            print 'Sig_a'+str(k)+':\n'+str(Sig_a[k])
          
          #print [(np.array([xt[j]-m_ML]).T,np.array([xt[j]-m_ML])) for j in xrange(nk)]
          #print [np.dot(np.array([xt[j]-m_ML]).T,np.array([xt[j]-m_ML])) for j in xrange(nk)]
          #print sum([np.dot(np.array([xt[j]-m_ML]).T,np.array([xt[j]-m_ML])) for j in xrange(nk)])
          #print ( float(k0a*nk)/kN ) * np.dot(np.array([m_ML - m0a]).T,np.array([m_ML - m0a]))
          #print VN
          #samp_sig_rand = np.array([ invwishart(nuN,VN) for i in xrange(100)])    ######
          #samp_sig = np.mean(samp_sig_rand,0)
          #print samp_sig
          
          #if np.linalg.det(samp_sig) < -0.0:
          #  Sig_a[k] = np.mean(np.array([ invwishartrand(nuN,VN)]),0)
          
          #print ''
          #for k in xrange(K):
          #if (nk[k] != 0):  #データなしは表示しない
          #  print 'Sig_a'+str(k)+':'+str(Sig_a[k])
          
        ########## ↑ ##### μa,Σa(ガウス分布の平均、共分散行列)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### μp,Σp(ガウス分布の平均、共分散行列)のサンプリング ##### ↓ ##########
        print u"Sampling myu_p,Sigma_p..."
        
        cc = np.sum([collections.Counter(zp[d]) for d in range(D)])
        for k in xrange(Kp) : 
          nk = cc[k]
          xt = []
          m_ML = np.zeros(dim_p)
          ###kについて、zaが同じものを集める
          if nk != 0 :  #もしzaの中にkがあれば(計算短縮処理)        ##0ワリ回避
            for d in xrange(D) : 
              for m in xrange(M[d]):
                if zp[d][m] == k : 
                  xt = xt + [ np.array(p_dm[d][m]) ]
            
            m_ML = sum(xt) / float(nk) #fsumではダメ
            #print "n:%d m_ML:%s" % (nk,str(m_ML))
            print "p%d n:%d" % (k,nk)
            
            ##ハイパーパラメータ更新
            kN = k0p + nk
            mN = ( k0p*m0p + nk*m_ML ) / kN  #dim_a 次元横ベクトル
            nN = n0p + nk
            VN = V0p + sum([np.dot(np.array([xt[j]-m_ML]).T,np.array([xt[j]-m_ML])) for j in xrange(nk)]) + (k0p*nk/kN)*np.dot(np.array([m_ML-m0p]).T,np.array([m_ML-m0p]))
            
            ##3.1##Σを逆ウィシャートからサンプリング
            Sig_p[k] = invwishart.rvs(df=nN, scale=VN) #/ n0a
            ##3.2##μを多変量ガウスからサンプリング
            Mu_p[k] = np.mean([multivariate_normal.rvs(mean=mN, cov=Sig_p[k]/kN) for i in xrange(100)],0) #サンプリングをロバストに
          else:  #データがないとき
            Mu_p[k]  = np.array([uniform(mu_p_init[0],mu_p_init[1]) for i in xrange(dim_p)])
            Sig_p[k] = invwishart.rvs(df=n0p, scale=V0p ) #np.eye(dim_p)*sig_p_init
          
          
          if (nk != 0):  #データなしは表示しない
            print 'Mu_p '+str(k)+' : '+str(Mu_p[k])
            print 'Sig_p'+str(k)+':\n'+str(Sig_p[k])
          
        ########## ↑ ##### μp,Σp(ガウス分布の平均、共分散行列)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### μo,Σo(ガウス分布の平均、共分散行列)のサンプリング ##### ↓ ##########
        print u"Sampling myu_o,Sigma_o..."
        
        cc = np.sum([collections.Counter(zo[d]) for d in range(D)])
        for k in xrange(Ko) : 
          nk = cc[k]
          xt = []
          m_ML = np.zeros(dim_o)
          ###kについて、zaが同じものを集める
          if nk != 0 :  #もしzaの中にkがあれば(計算短縮処理)        ##0ワリ回避
            for d in xrange(D) : 
              for m in xrange(M[d]):
                if zo[d][m] == k : 
                  xt = xt + [ np.array(o_dm[d][m]) ]
            
            m_ML = sum(xt) / float(nk) #fsumではダメ
            #print "n:%d m_ML:%s" % (nk,str(m_ML))
            print "o%d n:%d" % (k,nk)
            
            ##ハイパーパラメータ更新
            kN = k0o + nk
            mN = ( k0o*m0o + nk*m_ML ) / kN  #dim_a 次元横ベクトル
            nN = n0o + nk
            VN = V0o + sum([np.dot(np.array([xt[j]-m_ML]).T,np.array([xt[j]-m_ML])) for j in xrange(nk)]) + (k0o*nk/kN)*np.dot(np.array([m_ML-m0o]).T,np.array([m_ML-m0o]))
            
            ##3.1##Σを逆ウィシャートからサンプリング
            Sig_o[k] = invwishart.rvs(df=nN, scale=VN) #/ n0a
            ##3.2##μを多変量ガウスからサンプリング
            Mu_o[k] = np.mean([multivariate_normal.rvs(mean=mN, cov=Sig_o[k]/kN) for i in xrange(100)],0) #サンプリングをロバストに
          else:  #データがないとき
            Mu_o[k]  = np.array([uniform(mu_o_init[0],mu_o_init[1]) for i in xrange(dim_o)])
            Sig_o[k] = invwishart.rvs(df=n0o, scale=V0o ) #np.eye(dim_o)*sig_o_init
          
          if (nk != 0):  #データなしは表示しない
            print 'Mu_o '+str(k)+' : '+str(Mu_o[k])
            print 'Sig_o'+str(k)+':\n'+str(Sig_o[k])
          
        ########## ↑ ##### μo,Σo(ガウス分布の平均、共分散行列)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### μc,Σc(ガウス分布の平均、共分散行列)のサンプリング ##### ↓ ##########
        print u"Sampling myu_c,Sigma_c..."
        
        cc = np.sum([collections.Counter(zc[d]) for d in range(D)])
        for k in xrange(Kc) : 
          nk = cc[k]
          xt = []
          m_ML = np.zeros(dim_c)
          ###kについて、zaが同じものを集める
          if nk != 0 :  #もしzaの中にkがあれば(計算短縮処理)        ##0ワリ回避
            for d in xrange(D) : 
              for m in xrange(M[d]):
                if zc[d][m] == k : 
                  xt = xt + [ np.array(c_dm[d][m]) ]
            
            m_ML = sum(xt) / float(nk) #fsumではダメ
            #print "n:%d m_ML:%s" % (nk,str(m_ML))
            print "c%d n:%d" % (k,nk)
            
            ##ハイパーパラメータ更新
            kN = k0c + nk
            mN = ( k0c*m0c + nk*m_ML ) / kN  #dim_a 次元横ベクトル
            nN = n0c + nk
            VN = V0c + sum([np.dot(np.array([xt[j]-m_ML]).T,np.array([xt[j]-m_ML])) for j in xrange(nk)]) + (k0c*nk/kN)*np.dot(np.array([m_ML-m0c]).T,np.array([m_ML-m0c]))
            
            ##3.1##Σを逆ウィシャートからサンプリング
            Sig_c[k] = invwishart.rvs(df=nN, scale=VN) #/ n0a
            ##3.2##μを多変量ガウスからサンプリング
            Mu_c[k] = np.mean([multivariate_normal.rvs(mean=mN, cov=Sig_c[k]/kN) for i in xrange(100)],0) #サンプリングをロバストに
          else:  #データがないとき
            Mu_c[k]  = np.array([uniform(mu_c_init[0],mu_c_init[1]) for i in xrange(dim_c)])
            Sig_c[k] = invwishart.rvs(df=n0c, scale=V0c ) #np.eye(dim_c)*sig_c_init
          
          if (nk != 0):  #データなしは表示しない
            print 'Mu_c '+str(k)+' : '+str(Mu_c[k])
            print 'Sig_c'+str(k)+':\n'+str(Sig_c[k])
          
        ########## ↑ ##### μc,Σc(ガウス分布の平均、共分散行列)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### Fdのサンプリング ##### ↓ ##########
        print u"Sampling Fd..."
        #基本的にデータごとの単語数におけるモダリティの組み合わせ全探索する
        
        for d in xrange(D):
          F_temp = [f for f in itertools.permutations(modality,N[d])]  ##モダリティの順列組み合わせ
          temp = [1.0 for i in xrange(len(F_temp))]  ##フレームの組み合わせ数分の要素を用意する
          for i in xrange(len(F_temp)):
            for n in xrange(N[d]):
              #print i,n,N[d],M[d]
              if F_temp[i][n] == "a":
                temp[i] = temp[i] * theta[za[d]                   ][W_list.index(w_dn[d][n])]
              if F_temp[i][n] == "p":
                temp[i] = temp[i] * theta[zp[d][Ad[d]] + dict["p"]][W_list.index(w_dn[d][n])]
              if F_temp[i][n] == "o":
                temp[i] = temp[i] * theta[zo[d][Ad[d]] + dict["o"]][W_list.index(w_dn[d][n])]
              if F_temp[i][n] == "c":
                temp[i] = temp[i] * theta[zc[d][Ad[d]] + dict["c"]][W_list.index(w_dn[d][n])]
          temp = temp / np.sum(temp)  #正規化
          Fd[d] = F_temp[list(multinomial(1,temp)).index(1)]
          print d, Fd[d]
        
        ########## ↑ ##### Fdのサンプリング ##### ↑ ##########
        
        ########## ↓ ##### Θのサンプリング ##### ↓ ##########
        print u"Sampling Theta..."
        #dict = {"a":0, "p":Ka, "o":Ka+Kp, "c":Ka+Kp+Ko}   #各モダリティのindexにキーを足すとΘのindexになる
        
        #for i in xrange(L):
        temp = [np.array([gamma for w in xrange(W)]) for i in xrange(L)]
        for d in xrange(D):
            for n in xrange(N[d]):
              if (Fd[d][n] == "a"):
                temp[za[d]]                   [W_list.index(w_dn[d][n])] = temp[za[d]]                   [W_list.index(w_dn[d][n])] + 1
              if (Fd[d][n] == "p"):
                temp[zp[d][Ad[d]] + dict["p"]][W_list.index(w_dn[d][n])] = temp[zp[d][Ad[d]] + dict["p"]][W_list.index(w_dn[d][n])] + 1
              if (Fd[d][n] == "o"):
                temp[zo[d][Ad[d]] + dict["o"]][W_list.index(w_dn[d][n])] = temp[zo[d][Ad[d]] + dict["o"]][W_list.index(w_dn[d][n])] + 1
              if (Fd[d][n] == "c"):
                temp[zc[d][Ad[d]] + dict["c"]][W_list.index(w_dn[d][n])] = temp[zc[d][Ad[d]] + dict["c"]][W_list.index(w_dn[d][n])] + 1
          
        #加算したデータとパラメータから事後分布を計算しサンプリング
        theta = [sum(dirichlet(temp[i],100))/100.0 for i in xrange(L)] ##ロバストネスを上げる100
        
        print theta
        ########## ↑ ##### Θのサンプリング ##### ↑ ##########
        print ""  #改行用
      
      ######################################################################
      ####                       ↑学習フェーズ↑                       ####
      ######################################################################
      
      
      loop = 1
      ########  ↓ファイル出力フェーズ↓  ########
      if loop == 1:
        print "--------------------"
        #最終学習結果を出力
        print u"- <COMPLETED> Learning of Lexicon and Multiple Categories ver. iCub MODEL. -"
        #print 'Sample: ' + str(sample)
        print 'za: ' + str(za)
        print 'zp: ' + str(zp)
        print 'zo: ' + str(zo)
        print 'zc: ' + str(zc)
        for d in xrange(D):
          print 'Fd%d: %s' % (d, str(Fd[d]))
        for c in xrange(Ka):
          print "theta_a%d: %s" % (c,theta[c + dict["a"]])
        for c in xrange(Kp):
          print "theta_p%d: %s" % (c,theta[c + dict["p"]])
        for c in xrange(Ko):
          print "theta_o%d: %s" % (c,theta[c + dict["o"]])
        for c in xrange(Kc):
          print "theta_c%d: %s" % (c,theta[c + dict["c"]])
        for k in xrange(Ka):
          print "mu_a%d: %s" % (k, str(Mu_a[k]))
        for k in xrange(Kp):
          print "mu_p%d: %s" % (k, str(Mu_p[k]))
        for k in xrange(Ko):
          print "mu_o%d: %s" % (k, str(Mu_o[k]))
        for k in xrange(Kc):
          print "mu_c%d: %s" % (k, str(Mu_c[k]))
        #for k in xrange(K):
        #  print "sig%d: \n%s" % (k, str(Sig_a[k]))
        print 'pi_a: ' + str(pi_a)
        print 'pi_p: ' + str(pi_p)
        print 'pi_o: ' + str(pi_o)
        print 'pi_c: ' + str(pi_c)
        
        print "--------------------"
        
        #ファイルに保存
        para_save(foldername,trialname,filename,za,zp,zo,zc,Fd,theta,W_list,Mu_a,Sig_a,Mu_p,Sig_p,Mu_o,Sig_o,Mu_c,Sig_c,pi_a,pi_p,pi_o,pi_c)
        
      ########  ↑ファイル出力フェーズ↑  ########
      
      """
      ##学習後の描画用処理
      iti = []    #位置分布からサンプリングした点(x,y)を保存する
      #Plot = 500  #プロット数
      
      K_yes = 0
      ###全てのパーティクルに対し
      for j in range(K) : 
        yes = 0
        for t in xrange(N):  #jが推定された位置分布のindexにあるか判定
          if j == It[t]:
            yes = 1
        if yes == 1:
          K_yes = K_yes + 1
          for i in xrange(Plot):
            if (data_name != "test000"):
              S_temp = [[ S[j][0][0]/(0.05*0.05) , S[j][0][1]/(0.05*0.05) ] , [ S[j][1][0]/(0.05*0.05) , S[j][1][1]/(0.05*0.05) ]]
              x1,y1 = np.random.multivariate_normal( [(Myu[j][0][0][0]+37.8)/0.05, (Myu[j][1][0][0]+34.6)/0.05] , S_temp , 1).T
            else:
              x1,y1 = np.random.multivariate_normal([Myu[j][0][0][0],Myu[j][1][0][0]],S[j],1).T
            #print x1,y1
            iti = iti + [[x1,y1]]
      
      iti = iti + [[K_yes,Plot]]  #最後の要素[[位置分布の数],[位置分布ごとのプロット数]]
      #print iti
      filename2 = str(iteration) + "_" + str(sample)
      """
      
      #loop = 1 #メインループ用フラグ
      #while loop:
      #  #MAINCLOCK.tick(FPS)
      #  events = pygame.event.get()
      #  for event in events:
      #      if event.type == KEYDOWN:
      #          if event.key  == K_ESCAPE: exit()
      #  viewer.show(world,iti,0,[filename],[filename2])
      #  loop = 0
      
      
      




if __name__ == '__main__':
    import sys
    import shutil
    #import os.path
    from __init__ import *
    #from JuliusLattice_gmm import *
    #import time
    
    
    #filename = sys.argv[1]
    #print filename
    
    #出力ファイル名を要求
    #filename = raw_input("trialname?(folder) >")
    
    trialname = raw_input("trialname?(folder) >")
    start = raw_input("start number?>")
    end   = raw_input("end number?>")
    filename = raw_input("learning trial name?>")
    
    sn = int(start)
    en = int(end)
    Data = int(en) - int(sn) +1
    
    if D != Data:
      print "data number error.",D,Data
      exit()
    
    foldername = datafolder + trialname+"("+str(sn).zfill(3)+"-"+str(en).zfill(3)+")"
    
    #フォルダ作成
    Makedir( foldername + "/" + filename )
    Makedir( foldername + "/" + filename + "/init")
    
    #init.pyをコピー
    shutil.copy("./__init__.py", foldername + "/" + filename + "/init")
    
    #データの読み込み(単語、切り出した物体特徴と色と位置、姿勢と触覚と手先位置)
    M, N, w_dn, a_d, p_dm, o_dm, c_dm, Ad = data_read(trialname,filename,sn,en)
    #print w_dn
    
    #Gibbs sampling を実行
    simulate(foldername,trialname,filename,sn,en, M, N, w_dn, a_d, p_dm, o_dm, c_dm, Ad)
    
    
    
    
    
    
    
    #simulate(filename, data_read(filename))
    
    """
    for i in xrange(ITERATION):
      print "--------------------------------------------------"
      print "ITERATION:",i+1
      
      
      Julius_lattice(i,filename)    ##音声認識、ラティス形式出力、opemFST形式へ変換
      #p = os.popen( "python JuliusLattice_gmm.py " + str(i+1) +  " " + filename )
      
      
      
      while (os.path.exists("./data/" + filename + "/fst_gmm_" + str(i+1) + "/" + str(kyouji_count-1).zfill(3) +".fst" ) != True):
        print "./data/" + filename + "/fst_gmm_" + str(i+1) + "/" + str(kyouji_count-1).zfill(3) + ".fst",os.path.exists("./data/" + filename + "/fst_gmm_" + str(i+1).zfill(3) + "/" + str(kyouji_count-1) +".fst" ),"wait(60s)... or ERROR?"
        time.sleep(60.0) #sleep(秒指定)
      print "ITERATION:",i+1," Julius complete!"
      
      #for sample in xrange(sample_num):
      sample = 0  ##latticelmのパラメータ通りだけサンプルする
      for p1 in xrange(len(knownn)):
        for p2 in xrange(len(unkn)):
          if sample < sample_num:
            print "latticelm run. sample_num:" + str(sample)
            p = os.popen( "latticelm -input fst -filelist data/" + filename + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile data/" + filename + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2]) )   ##latticelm  ## -annealsteps 10 -anneallength 15
            time.sleep(1.0) #sleep(秒指定)
            while (os.path.exists("./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100" ) != True):
              print "./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100",os.path.exists("./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100" ),"wait(30s)... or ERROR?"
              p.close()
              p = os.popen( "latticelm -input fst -filelist data/" + filename + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile data/" + filename + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2]) )   ##latticelm  ## -annealsteps 10 -anneallength 15
              
              time.sleep(30.0) #sleep(秒指定)
            sample = sample + 1
            p.close()
      print "ITERATION:",i+1," latticelm complete!"
      
      simulate(i+1)          ##場所概念の学習
      
      print "ITERATION:",i+1," Learning complete!"
      sougo(i+1)             ##相互情報量計算+##単語辞書登録
      print "ITERATION:",i+1," Language Model update!"
      #Language_model_update(i+1)  ##単語辞書登録
    """
    ##ループ後処理
    
    #p0.close()
    





########################################



















"""
def sougo(iteration):
  #MI_Samp = [0.0 for sample in xrange(sample_num)]  ##サンプルの数だけMIを求める
  MI_Samp2 = [0.0 for sample in xrange(sample_num)]  ##サンプルの数だけMIを求める
  #tanjyun_log = [0.0 for sample in xrange(sample_num)]
  #tanjyun_log2 = [0.0 for sample in xrange(sample_num)]
  #N = 0      #データ個数用
  #sample_num = 1  #取得するサンプル数
  Otb_Samp = [[] for sample in xrange(sample_num)]   #単語分割結果：教示データ
  W_index = [[] for sample in xrange(sample_num)]
  
  for sample in xrange(sample_num):
    
    #####↓##発話した文章ごとに相互情報量を計算し、サンプリング結果を選ぶ##↓######
    
    ##発話認識文データを読み込む
    ##空白またはカンマで区切られた単語を行ごとに読み込むことを想定する
    
    N = 0
    #for sample in xrange(sample_num):
    #テキストファイルを読み込み
    for line in open('./data/' + filename + '/out_gmm_' + str(iteration) + '/' + str(sample) + '_samp.100', 'r'):   ##*_samp.100を順番に読み込む
        itemList = line[:-1].split(' ')
        
        #<s>,<sp>,</s>を除く処理：単語に区切られていた場合
        for b in xrange(5):
          if ("<s><s>" in itemList):
            itemList.pop(itemList.index("<s><s>"))
          if ("<s><sp>" in itemList):
            itemList.pop(itemList.index("<s><sp>"))
          if ("<s>" in itemList):
            itemList.pop(itemList.index("<s>"))
          if ("<sp>" in itemList):
            itemList.pop(itemList.index("<sp>"))
          if ("<sp><sp>" in itemList):
            itemList.pop(itemList.index("<sp><sp>"))
          if ("</s>" in itemList):
            itemList.pop(itemList.index("</s>"))
          if ("<sp></s>" in itemList):
            itemList.pop(itemList.index("<sp></s>"))
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        #<s>,<sp>,</s>を除く処理：単語中に存在している場合
        for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("<s><s>", "")
          itemList[j] = itemList[j].replace("<s>", "")
          itemList[j] = itemList[j].replace("<sp>", "")
          itemList[j] = itemList[j].replace("</s>", "")
        for b in xrange(5):
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        
        Otb_Samp[sample] = Otb_Samp[sample] + [itemList]
        #if sample == 0:
        N = N + 1  #count
        
        #for j in xrange(len(itemList)):
        #    print u"%s " % (str(itemList[j])),
        #print u""  #改行用
        
        
    
    ##場所の名前の多項分布のインデックス用
    #W_index = []
    #for sample in xrange(sample_num):    #サンプル個分
    for n in xrange(N):                #発話文数分
        for j in xrange(len(Otb_Samp[sample][n])):   #一文における単語数分
          if ( (Otb_Samp[sample][n][j] in W_index[sample]) == False ):
            W_index[sample].append(Otb_Samp[sample][n][j])
            #print str(W_index),len(W_index)
    
    print "(",
    for i in xrange(len(W_index[sample])):
      print "\""+ str(i) + ":" + str(W_index[sample][i]) + "\",",  #unicode(W_index[sample][i], 'shift-jis').encode('utf-8')
    print ")"
    
    
    #print type(W_index[sample][i])
    #print type(unicode(W_index[sample][i], 'shift-jis').encode('utf-8'))
    #print type(unicode(W_index[sample][i], 'utf-8'))
    
  ##サンプリングごとに、時刻tデータごとにBOW化(?)する、ベクトルとする
  Otb_B_Samp = [ [ [] for n in xrange(N) ] for ssss in xrange(sample_num) ]
  for sample in xrange(sample_num):
    for n in xrange(N):
      Otb_B_Samp[sample][n] = [0 for i in xrange(len(W_index[sample]))]
  
  for sample in xrange(sample_num):
    #for sample in xrange(sample_num):
    for n in xrange(N):
      for j in xrange(len(Otb_Samp[sample][n])):
          #print n,j,len(Otb_Samp[sample][n])
          for i in xrange(len(W_index[sample])):
            if (W_index[sample][i] == Otb_Samp[sample][n][j] ):
              Otb_B_Samp[sample][n][i] = Otb_B_Samp[sample][n][i] + 1
    #print Otb_B
    
    
    
    W = [ [beta0 for j in xrange(len(W_index[sample]))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
    pi = [ 0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
    #Ct = [ int(uniform(0,L)) for n in xrange(N)]
    Ct = []
    
    ##piの読み込み
    for line in open('./data/' + filename +'/' + filename + '_pi_'+str(iteration) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            pi[i] = float(itemList[i])
        
    ##Ctの読み込み
    for line in open('./data/' + filename +'/' + filename + '_Ct_'+str(iteration) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            Ct = Ct + [int(itemList[i])]
        
    ##Wの読み込み
    c = 0
    #テキストファイルを読み込み
    for line in open('./data/' + filename +'/' + filename + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              W[c][i] = float(itemList[i])
              
              #print itemList
        c = c + 1
    
    
    
    #####↓##場所概念ごとに単語ごとに相互情報量を計算、高いものから表示##↓######
    ##相互情報量による単語のセレクション
    MI = [[] for c in xrange(L)]
    W_in = []    #閾値以上の単語集合
    #W_out = []   #W_in以外の単語
    #i_best = len(W_index)    ##相互情報量上位の単語をどれだけ使うか
    #MI_best = [ ['' for c in xrange(L)] for i in xrange(i_best) ]
    ###相互情報量を計算
    for c in xrange(L):
      #print "Concept:%d" % c
      #W_temp = Multinomial(W[c])
      for o in xrange(len(W_index[sample])):
        word = W_index[sample][o]
        
        ##BOW化(?)する、ベクトルとする
        #Otb_B = [0 for i in xrange(len(W_index[sample]))]
        #Otb_B[o] = 1
        
        #print W[c]
        #print Otb_B
        
        score = MI_binary(o,W,pi,c)
        
        MI[c].append( (score, word) )
        
        if (score >= threshold):  ##閾値以上の単語をすべて保持
          #print score , threshold ,word in W_in
          if ((word in W_in) == False):  #リストに単語が存在しないとき
            #print word
            W_in = W_in + [word]
        #else:
        #  W_out = W_out + [word]
        
      MI[c].sort(reverse=True)
      #for i in xrange(i_best):
      #  MI_best[i][c] = MI[c][i][1]
      
      #for score, word in MI[c]:
      #  print score, word
    
    ##ファイル出力
    fp = open('./data/' + filename + '/' + filename + '_sougo_C_' + str(iteration) + '_' + str(sample) + '.csv', 'w')
    for c in xrange(L):
      fp.write("Concept:" + str(c) + '\n')
      for score, word in MI[c]:
        fp.write(str(score) + "," + word + '\n')
      fp.write('\n')
    #for c in xrange(len(W_index)):
    fp.close()
    
    #####↑##場所概念ごとに単語ごとに相互情報量を計算、高いものから表示##↑######
    
    if (len(W_in) == 0 ):
      print "W_in is empty."
      W_in = W_index[sample] ##選ばれる単語がなかった場合、W_indexをそのままいれる
    
    print W_in
    
    ##場所の名前W（多項分布）をW_inに含まれる単語のみにする
    
    #for j in xrange(len(W_index[sample])):
    #  for i in xrange(len(W_in)):
    #    if (W_index[sample][j] in W_in == False):
    #      W_out = W_out + [W_index[sample][j]]
    
    
    W_reco = [ [0.0 for j in xrange(len(W_in))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
    #W_index_reco = ["" for j in xrange(len(W_in))]
    #Otb_B_Samp_reco = [ [0 for j in xrange(len(W_in))] for n in xrange(N) ]
    #print L,N
    #print W_reco
    for c in xrange(L):
      for j in xrange(len(W_index[sample])):
        for i in xrange(len(W_in)):
          if ((W_in[i] in W_index[sample][j]) == True):
            W_reco[c][i] = float(W[c][j])
            #for t in xrange(N):
            #  Otb_B_Samp_reco[t][i] = Otb_B_Samp[sample][t][j]
      
      #正規化処理
      W_reco_sum = fsum(W_reco[c])
      W_reco_max = max(W_reco[c])
      W_reco_summax = float(W_reco_sum) / W_reco_max
      for i in xrange(len(W_in)):
        W_reco[c][i] = float(float(W_reco[c][i])/W_reco_max) / W_reco_summax
    
    #print W_reco
    
    ###相互情報量を計算(それぞれの単語とCtとの共起性を見る)
    MI_Samp2[sample] = Mutual_Info(W_reco,pi)
    
    print "sample:",sample," MI:",MI_Samp2[sample]
    
    
  MAX_Samp = MI_Samp2.index(max(MI_Samp2))  #相互情報量が最大のサンプル番号
  
  ##ファイル出力
  fp = open('./data/' + filename + '/' + filename + '_sougo_MI_' + str(iteration) + '.csv', 'w')
  #fp.write(',Samp,Samp2,tanjyun_log,tanjyun_log2,' +  '\n')
  for sample in xrange(sample_num):
      fp.write(str(sample) + ',' + str(MI_Samp2[sample]) + '\n') 
      #fp.write(str(sample) + ',' + str(MI_Samp[sample]) + ',' + str(MI_Samp2[sample]) + ',' + str(tanjyun_log[sample])  + ',' + str(tanjyun_log2[sample]) + '\n')  #文章ごとに計算
  fp.close()
  
  #  #####↑##発話した文章ごとに相互情報量を計算し、サンプリング結果を選ぶ##↑######
  
  #def Language_model_update(iteration):
  #""#"
  #  ###推定された場所概念番号を調べる
  #  L_dd = [0 for c in xrange(L)]
  #  for t in xrange(len(Ct)):
  #    for c in xrange(L):
  #      if Ct[t] == c:
  #        L_dd[c] = 1
  #  ##print L_dd #ok
  #""#"
  
  ###↓###単語辞書読み込み書き込み追加############################################
  LIST = []
  LIST_plus = []
  i_best = len(W_index[MAX_Samp])    ##相互情報量上位の単語をどれだけ使うか（len(W_index)：すべて）
  W_index = W_index[MAX_Samp]
  hatsuon = [ "" for i in xrange(i_best) ]
  TANGO = []
  ##単語辞書の読み込み
  for line in open('./lang_m/' + lang_init, 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("(", "")
          itemList[j] = itemList[j].replace(")", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
      
      
  #print TANGO
  
  #dd_num = 0
  ##W_indexの単語を順番に処理していく
  #for i in xrange(len(W_index)):
  #  W_index_sj = unicode(W_index[i], encoding='shift_jis')
  #for i in xrange(L):
  #  if L_dd[i] == 1:
  for c in xrange(i_best):    # i_best = len(W_index)
          #W_index_sj = unicode(MI_best[c][i], encoding='shift_jis')
          W_index_sj = unicode(W_index[c], encoding='shift_jis')
          if len(W_index_sj) != 1:  ##１文字は除外
            #for moji in xrange(len(W_index_sj)):
            moji = 0
            while (moji < len(W_index_sj)):
              flag_moji = 0
              #print len(W_index_sj),str(W_index_sj),moji,W_index_sj[moji]#,len(unicode(W_index[i], encoding='shift_jis'))
              
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-2 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+"_"+W_index_sj[moji+2]) and (W_index_sj[moji+1] == "_"): 
                    print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+W_index_sj[moji+1]):
                    print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_index_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_index_sj) > moji) and (flag_moji == 0):
                  #else:
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                      print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
            print hatsuon[c]
          else:
            print W_index[c] + " (one name)"
        
  
  ##各場所の名前の単語ごとに
  
  meishi = u'名詞'
  meishi = meishi.encode('shift-jis')
  
  ##単語辞書ファイル生成
  fp = open('./data/' + filename + '/web.000s_' + str(iteration) + '.htkdic', 'w')
  for list in xrange(len(LIST)):
        fp.write(LIST[list])
  #fp.write('\n')
  #for c in xrange(len(W_index)):
  ##新しい単語を追加
  #i = 0
  c = 0
  #while i < L:
  #  #if L_dd[i] == 1:
  for mi in xrange(i_best):    # i_best = len(W_index)
        if hatsuon[mi] != "":
            if ((W_index[mi] in LIST_plus) == False):  #同一単語を除外
              flag_tango = 0
              for j in xrange(len(TANGO)):
                if(W_index[mi] == TANGO[j][0]):
                  flag_tango = -1
              if flag_tango == 0:
                LIST_plus = LIST_plus + [W_index[mi]]
                
                fp.write(LIST_plus[c] + "+" + meishi +"	[" + LIST_plus[c] + "]	" + hatsuon[mi])
                fp.write('\n')
                c = c+1
  #i = i+1
  fp.close()
  
  ###↑###単語辞書読み込み書き込み追加############################################
  
"""
