#coding:utf-8

##############################################
# SpCoSLAM online learning program (無駄なコメントアウトコードを省いたバージョン+bugfix)
# かつ、SpCoAのオンライン版（SpCoSLAMから言語モデル更新と画像特徴を除けるバージョン）+パーティクルごとにLM
# Akira Taniguchi 2017/01/18-2017/02/02-2017/02/15-2017/02/21
##############################################

##########---処理の流れ（gmapping側）---##########

#これまでのすべての時刻のパーティクル情報（index,自己位置座標、重み、前時刻のindex）を毎フレーム、ファイル出力

#教示時刻のフラグが立つ
##rosbagを一時停止
##フラグ処理
###重みのファイルが読み込めるまでリサンプリングのコードまで進まない
###読み込めたら重み更新して次へ進む
###フラグを切る
##rosbagを再開


##########---処理の流れ（Pythonプログラム側）---##########
#教示時刻のフラグよりPythonプログラムが呼び出される

###（この処理は前回の単語辞書が得られる時点で先に裏で計算を回しておくのもあり。）
#JuliusLattice_gmm.pyを呼び出す
##前回の言語モデルを読み込み
##Juliusで教示時刻までの教示データに対して音声認識
##音声認識結果（ラティス）のファイル形式の変換
##latticelmで単語分割（パーティクル個数回）

#画像特徴ftを読み込み（事前に全教示時刻のCNN特徴を得ておく）

#gmapping側が出力した情報の読み込み
##現在時刻のパーティクル情報（index,自己位置座標、重み、前時刻のindex）を取得（ファイル読み込み）

#過去の教示時刻のパーティクル情報の読み込み
##パーティクルの時刻ごとのindex対応付け処理（前回の教示のどのパーティクルIDが今回のIDなのか）
##x_{0:t}を情報整理して得る

#パーティクルごとに計算
##単語情報S_{1:t}、過去のカウント情報C_{1:t-1},i_{1:t-1}、n(l,k),n(l,g),n(l,e),n(k)をファイル読み込み
##it,Ctをサンプリング
### 単語、画像特徴、場所概念index、位置分布indexのカウント数の計算
### スチューデントt分布の計算
##画像の重みwfを計算（サンプリング時に計算済み）
##単語の重みwsを計算（サンプリング時に計算済み）
##重みの掛け算wt=wz*wf*ws

#重みの正規化wt
#パーティクルIDごとにwtをファイル保存

#パーティクルIDごとに単語情報S_{1:t}、カウント情報C_{1:t},i_{1:t}、n(l,k),n(l,g),n(l,e),n(k)をファイル書き込み

#最大重みのパーティクルの単語情報から単語辞書作成


##########---遂行タスク---##########
###Fast SLAMの方の重みが強く効く可能性が高い。→できるだけ毎回リサンプリング？

#ファイル構造の整理

##########---作業終了タスク---##########
#latticelm のオプションコマンドを__init__.pyで編集できるようにした。
#パーティクルの構造体を定義
##スチューデントt分布の計算(pdf,logpdf)
##最大重みのパーティクルindexを保存する
#このプログラムを呼び出すプログラムの作成（trialnameの受け渡し、フォルダ作成）
###実際のindex番号と処理上のindex番号の対応付けを保存（場所概念パラメータΘは処理場の順番）
##Wとθの計算のbug fix (2017/02/12)
#ST,FTの新規クラス時のカウント計算bug fix (2017/02/13)
#Btの計算を修正
#CTとITのパーティクルＩＤ対応付けbug fix (2017/02/15)
#piとphiの推定値の計算修正 (2017/02/21)

###(動作未確認) -> bug fix
#JuliusLattice_gmm.pyの編集
#ParticleSearcher関数
#m_count2step関数
#ReadParticleData関数  #パーティクル情報の取得（ファイル読み込み）
#ReadWordData関数
#ReaditCtData関数
#ReadImageData
#WordDictionaryUpdate関数
#WriteParticleData
#WriteWordData
#SaveParameters
#Learning関数

##########---保留---##########



##############################################
import os
import re
import glob
import random
import collections
import numpy as np
import scipy as sp
from numpy.random import multinomial#,uniform #,dirichlet
from scipy.stats import t#,multivariate_normal,invwishart,rv_discrete
from numpy.linalg import inv, cholesky
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2,gamma,lgamma
#from sklearn.cluster import KMeans
from __init__ import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass

def multivariate_t_distribution(x, mu, Sigma, df):
    """
    Multivariate t-student density. Returns the density
    of the function at points specified by x.
    
    input:
        x = parameter (n-d numpy array; will be forced to 2d)
        mu = mean (d dimensional numpy array)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        
    Edited from: http://stackoverflow.com/a/29804411/3521179
    """
    
    x = np.atleast_2d(x) # requires x as 2d
    nD = Sigma.shape[0] # dimensionality
    
    numerator = gamma(1.0 * (nD + df) / 2.0)
    denominator = (
            gamma(1.0 * df / 2.0) * 
            np.power(df * PI, 1.0 * nD / 2.0) *  
            np.power(np.linalg.det(Sigma), 1.0 / 2.0) * 
            np.power(
                1.0 + (1.0 / df) *
                np.diagonal(
                    np.dot( np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)
                ), 
                1.0 * (nD + df) / 2.0
                )
            )
            
    return 1.0 * numerator / denominator 

def log_multivariate_t_distribution(x, mu, Sigma, df):
    x = np.atleast_2d(x) # requires x as 2d
    nD = Sigma.shape[0] # dimensionality
    
    lnumerator = lgamma( (nD + df) / 2.0 )
    ldenominator = (
            lgamma(0.5 * df) + 
            (0.5 * nD) * ( log(df) + log(PI) ) + 
            0.5 * log(np.linalg.det(Sigma))  + 
            (0.5 * (nD + df)) * 
            log(1.0 + (1.0 / df) * np.diagonal(np.dot( np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)))
            )
            
    return lnumerator - ldenominator 


# Reading data for image feature(未使用)
def ReadImageData(trialname, datasetname, step):
  FT = []
  for s in xrange(step):
    for line in open( datasetfolder + datasetname + 'img/ft' + str(s+1) + '.csv', 'r'):
      itemList = line[:].split(',')
    FT.append( [float(itemList[i]) for i in xrange(DimImg)] )
  
  return FT


#時刻情報を取得して、一番近い画像特徴ファイルを読み込むようにする
def ReadImageData2(trialname, datasetname, step, clocktime):
  clocktime = int(clocktime)
  
  FT = []
  for s in xrange(step-1):
    for line in open( datafolder + trialname + '/img/ft' + str(s+1) + '.csv', 'r'):
      itemList = line[:].split(',')
    FT.append( [float(itemList[i]) for i in xrange(DimImg)] )
  
  #現在時刻の画像データを読み込む
  files = glob.glob(datasetfolder + datasetname + Descriptor + "*.csv")
  files.sort()
  files2 = [files[i][len(datasetfolder)+len(datasetname)+len(Descriptor)+1:-4] for i in xrange(len(files))]
  #print files2
  #temptime = []
  #for f in files:
  #  m = re.match('[^\d]*(\d+).*$', f)
  #  temptime.append(int(m.groups[0]))
  temptime = [int(re.match('[^\d]*(\d+).*$', f).groups()[0]) for f in files2]
  if (clocktime in temptime):
    f_temp = files[temptime.index(clocktime)]
  else:
    if (clocktime+1 in temptime):
      f_temp = files[temptime.index(clocktime)]
    elif(clocktime-1 in temptime):
      f_temp = files[temptime.index(clocktime)]
    else:
      print "FT read error.",clocktime,temptime
      f_temp = files[-1]
  
  for line in open( f_temp, 'r'):
      itemList = line[:].split(',')
  FT.append( [float(itemList[i]) for i in xrange(DimImg)] )
  
  #新規データをファイル書き込み
  fp = open( datafolder + trialname + '/img/ft' + str(step) + '.csv', 'w')
  for i in xrange(DimImg):
    fp.write(repr(float(itemList[i]))+',')
  fp.close()
  
  #画像ファイルの時刻情報を保存
  fp = open( datafolder + trialname + '/img/time' + str(step) + '.csv', 'w')
  fp.write(repr(f_temp))
  fp.close()
  
  return FT

# Reading word data and Making word list
def ReadWordData(step, filename, particle):
      N = 0
      Otb = []
      #テキストファイルを読み込み
      for line in open( filename + '/out_gmm/' + str(particle) + '_samp.'+str(samps), 'r'):   ##*_samp.*を読み込む
        itemList = line[:-1].split(' ')
        
        #<s>,<sp>,</s>を除く処理：単語に区切られていた場合
        #for b in xrange(5):
        while ("<s><s>" in itemList):
            itemList.pop(itemList.index("<s><s>"))
        while ("<s><sp>" in itemList):
            itemList.pop(itemList.index("<s><sp>"))
        while ("<s>" in itemList):
            itemList.pop(itemList.index("<s>"))
        while ("<sp>" in itemList):
            itemList.pop(itemList.index("<sp>"))
        while ("<sp><sp>" in itemList):
            itemList.pop(itemList.index("<sp><sp>"))
        while ("</s>" in itemList):
            itemList.pop(itemList.index("</s>"))
        while ("<sp></s>" in itemList):
            itemList.pop(itemList.index("<sp></s>"))
        while ("" in itemList):
            itemList.pop(itemList.index(""))
        #<s>,<sp>,</s>を除く処理：単語中に存在している場合
        for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("<s><s>", "")
          itemList[j] = itemList[j].replace("<s>", "")
          itemList[j] = itemList[j].replace("<sp>", "")
          itemList[j] = itemList[j].replace("</s>", "")
        
        #for b in xrange(5):
          # if ("" in itemList):
        while ("" in itemList):
          itemList.pop(itemList.index(""))
        
        #Otb[sample] = Otb[sample] + [itemList]
        Otb = Otb + [itemList]
        N = N + 1  #count
        
        for j in xrange(len(itemList)):
            print "%s " % (unicode(itemList[j], encoding='shift_jis')),
        print ""  #改行用
      #unicode(W_list[c], encoding='shift_jis')
      
      ##場所の名前の多項分布のインデックス用
      W_list = []
      for n in xrange(N):
        for j in xrange(len(Otb[n])):
          if ( (Otb[n][j] in W_list) == False ):
            W_list.append(Otb[n][j])
            #print str(W_list),len(W_list)
      
      ##時刻tデータごとにBOW化(?)する、ベクトルとする
      Otb_BOW = [ [0 for i in xrange(len(W_list))] for n in xrange(N) ]
      
      for n in xrange(N):
        for j in xrange(len(Otb[n])):
          for i in xrange(len(W_list)):
            if ( W_list[i] == Otb[n][j] ):
              Otb_BOW[n][i] = Otb_BOW[n][i] + 1
      
      return W_list, Otb_BOW


#itとCtのデータを読み込む（教示した時刻のみ）
def ReaditCtData(trialname, step, particle):
  CT,IT = [],[]
  if (step != 1):  #最初のステップ以外
    for line in open( datafolder + trialname + "/" + str(step-1) + "/particle" + str(particle) + ".csv" , 'r' ):
      itemList = line[:-1].split(',')
      CT.append( int(itemList[7]) )
      IT.append( int(itemList[8]) )
  return CT, IT

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
    flag = 0
    fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
    fp.write(str(flag))
    fp.close()
    exit()
  
  #教示された時刻のみのデータにする
  steplist = m_count2step(trialname, m_count)
  step = len(steplist)
  #print steplist
  
  #C[1:t-1],I[1:t-1]のパーティクルID(step-1時点)と現在のparticleIDの対応付け
  CTtemp = [[] for r in xrange(R)]
  ITtemp = [[] for r in xrange(R)]
  for particle in xrange(R):
    CTtemp[particle],ITtemp[particle] = ReaditCtData(trialname, step, particle)
  
  p = [[] for c in xrange(m_count)]
  for c in xrange(m_count):
    p[c] = ReadParticleData(c+1, trialname)        #m_countのindexは1から始まる   
    ######非効率なので、前回のパーティクル情報を使う（未実装）
  
  p_trajectory = [ [0.0 for c in xrange(m_count)] for i in xrange(R) ]
  CT = [ [0 for s in xrange(step-1)] for i in xrange(R) ]
  IT = [ [0 for s in xrange(step-1)] for i in xrange(R) ]
  
  for i in xrange(R):
    c_count = m_count-1  #一番最後の配列から処理
    #print c_count,i
    p_trajectory[i][c_count] = p[c_count][i]
    for c in xrange(m_count-1):  #0～最後から2番目の配列まで
      preID = p[c_count][p_trajectory[i][c_count].id].pid
      p_trajectory[i][c_count-1] = p[c_count-1][preID]
      if (step == 1):
        CT[i] = CTtemp[i]
        IT[i] = ITtemp[i]
      elif (step == 2):
        CT[i] = [1]
        IT[i] = [1]
      else:
        if (steplist[-2][0] == c_count): #CTtemp,ITtempを現在のパーティクルID順にする
          CT[i] = [ CTtemp[preID][s] for s in xrange(step-1)]
          IT[i] = [ ITtemp[preID][s] for s in xrange(step-1)]
          #print i,preID
      #print i, c, c_count-1, preID
      c_count -= 1
  
  X_To = [ [0.0 for c in xrange(step)] for i in xrange(R) ]
  for i in xrange(R):
    X_To[i] = [p_trajectory[i][steplist[s][0]-1] for s in xrange(step)]
  
  return X_To, step, m_count, CT, IT


#gmappingの時刻カウント数（m_count）と教示時刻のステップ数（step）を対応付ける
def m_count2step(trialname, m_count):
  list= []  #[ [m_count, step], ... ]
  step = 1
  csvname = datafolder + trialname + "/m_count2step.csv"
  
  if ( os.path.exists( csvname ) != True ):  ##ファイルがないとき、作成する
    fp = open( csvname, 'w')
    fp.write("")
    fp.close()
  else:
    for line in open ( csvname , 'r'):
      itemList = line[:-1].split(',')
      #print itemList
      list.append( [int(itemList[0]), int(itemList[1])] )
      step += 1
  
  
  #Update m_count2step.csv
  if (step == len(list)+1) or (step == 1 and len(list) == 1):  #テスト用の実行で同じm_countのデータカウントが増えないように
    fp = open( csvname, 'a')
    fp.write(str(m_count) + "," + str(step))
    fp.write('\n')
    fp.close()
    list.append( [m_count, step] )
  
  return list


#パーティクル情報の保存
def WriteParticleData(filename, step, particle, Xp, p_weight, ct, it, CT, IT):
  cstep = step - 1
  #ID,x,y,theta,weight,pID,Ct,it
  fp = open( filename + "/particle" + str(particle) + ".csv", 'w')
  for s in xrange(step-1):
      fp.write( str(s) + "," + str(Xp[s].id) + "," + str(Xp[s].x) + "," + str(Xp[s].y) + "," + str(Xp[s].theta) + "," + str(Xp[s].weight) + "," + str(Xp[s].pid) + "," + str(CT[s]) + "," + str(IT[s]) )
      fp.write('\n')
  fp.write( str(cstep) + "," + str(Xp[cstep].id) + "," + str(Xp[cstep].x) + "," + str(Xp[cstep].y) + "," + str(Xp[cstep].theta) + "," + str(p_weight) + "," + str(Xp[cstep].pid) + "," + str(ct) + "," + str(it) )
  fp.write('\n')

#パーティクルごとに単語情報を保存
def WriteWordData(filename, particle, W_list_i):
  fp = open( filename + "/W_list" + str(particle) + ".csv", 'w')
  for w in xrange(len(W_list_i)):
    fp.write(W_list_i[w]+",")
  fp.close()

#重み（log）を保存（gmapping読み込み用）
def WriteWeightData(trialname, m_count, p_weight_log):
  fp = open( datafolder + trialname + "/weight/" + str(m_count) + ".csv", 'w')
  for r in xrange(R):
    fp.write(str(p_weight_log[r])+",")
  fp.close()

#パーティクルごとのCtとitのインデックス対応を保存
def WriteIndexData(filename, particle, ccitems, icitems,ct,it):
  fp = open( filename + "/index" + str(particle) + ".csv", 'w')
  for c in xrange(len(ccitems)):
    fp.write(str(ccitems[c][0])+",")
  if ( ct == (max(ccitems)[0]+1) ):
    fp.write(str(ct) + ",")
  fp.write("\n")
  for i in xrange(len(icitems)):
    fp.write(str(icitems[i][0])+",")
  if ( it == (max(icitems)[0]+1) ):
    fp.write(str(it) + ",")
  fp.write("\n")
  fp.close()

# Saving data for parameters Θ of spatial concepts
def SaveParameters(filename, particle, phi, pi, W, theta, mu, sig):
  fp = open( filename + "/phi" + str(particle) + ".csv", 'w')
  for i in xrange(len(phi)):
    for j in xrange(len(phi[i])):
      fp.write(repr(phi[i][j])+",")
    fp.write('\n')
  fp.close()
  
  fp2 = open( filename + "/pi" + str(particle) + ".csv", 'w')
  for i in xrange(len(pi)):
    fp2.write(repr(pi[i])+",")
  fp2.write('\n')
  fp2.close()
  
  fp3 = open( filename + "/W" + str(particle) + ".csv", 'w')
  for i in xrange(len(W)):
    for j in xrange(len(W[i])):
      fp3.write(repr(W[i][j])+",")
    fp3.write('\n')
  fp3.close()
  
  fp4 = open( filename + "/theta" + str(particle) + ".csv", 'w')
  for i in xrange(len(theta)):
    for j in xrange(len(theta[i])):
      fp4.write(repr(theta[i][j])+",")
    fp4.write('\n')
  fp4.close()
  
  fp5 = open(filename + "/mu" + str(particle) + ".csv", 'w')
  for k in xrange(len(mu)):
      for dim in xrange(len(mu[k])):
        fp5.write(repr(mu[k][dim])+',')
      fp5.write('\n')
  fp5.close()
  
  fp6 = open(filename + "/sig" + str(particle) + ".csv", 'w')
  for k in xrange(len(sig)):
      for dim in xrange(dimx):
        for dim2 in xrange(dimx):
          fp6.write(repr(sig[k][dim][dim2])+',')
        #fp6.write('\n')
      fp6.write('\n')
  fp6.close()


###↓###単語辞書読み込み書き込み追加############################################
#MAX_Samp : 重みが最大のパーティクル
def WordDictionaryUpdate(step, filename, W_list, particle):
  LIST = []
  LIST_plus = []
  #i_best = len(W_list[MAX_Samp])    ##相互情報量上位の単語をどれだけ使うか（len(W_list)：すべて）
  i_best = len(W_list)
  #W_list = W_list[MAX_Samp]
  hatsuon = [ "" for i in xrange(i_best) ]
  TANGO = []
  ##単語辞書の読み込み
  for line in open('./lang_m/' + lang_init, 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
      
  #print TANGO
  if (UseLM == 1):
    ##W_listの単語を順番に処理していく
    for c in xrange(i_best):    # i_best = len(W_list)
          #W_list_sj = unicode(MI_best[c][i], encoding='shift_jis')
          W_list_sj = unicode(W_list[c], encoding='shift_jis')
          if len(W_list_sj) != 1:  ##１文字は除外
            #for moji in xrange(len(W_list_sj)):
            moji = 0
            while (moji < len(W_list_sj)):
              flag_moji = 0
              #print len(W_list_sj),str(W_list_sj),moji,W_list_sj[moji]#,len(unicode(W_list[i], encoding='shift_jis'))
              
              for j in xrange(len(TANGO)):
                if (len(W_list_sj)-2 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]+"_"+W_list_sj[moji+2]) and (W_list_sj[moji+1] == "_"): 
                    ###print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_list_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]+W_list_sj[moji+1]):
                    ###print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_list_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_list_sj) > moji) and (flag_moji == 0):
                  #else:
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]):
                      ###print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
            print hatsuon[c]
          else:
            print W_list[c] + " (one name)"
  
  ##各場所の名前の単語ごとに
  meishi = u'名詞'
  meishi = meishi.encode('shift-jis')
  
  ##単語辞書ファイル生成
  fp = open( filename + '/WD'+str(particle)+'.htkdic', 'w')
  for list in xrange(len(LIST)):
        fp.write(LIST[list])
  if (UseLM == 1):
    ##新しい単語を追加
    c = 0
    for mi in xrange(i_best):    # i_best = len(W_list)
        if hatsuon[mi] != "":
            if ((W_list[mi] in LIST_plus) == False):  #同一単語を除外
              flag_tango = 0
              for j in xrange(len(TANGO)):
                if(W_list[mi] == TANGO[j][0]):
                  flag_tango = -1
              if flag_tango == 0:
                LIST_plus = LIST_plus + [W_list[mi]]
                
                fp.write(LIST_plus[c] + "+" + meishi +"	[" + LIST_plus[c] + "]	" + hatsuon[mi])
                fp.write('\n')
                c = c+1
  
  fp.close()
  ###↑###単語辞書読み込み書き込み追加############################################


# Online Learning for Spatial Concepts of one particle
def Learning(step, filename, particle, XT, ST, W_list, CT, IT, FT):
    np.random.seed()
    ######################################################################
    ####                       ↓学習フェーズ↓                       ####
    ######################################################################
    print u"- <START> Learning of Spatial Concepts in Particle:" + str(particle) + " -"
    ##sampling ct and it
    print u"Sampling Ct,it..."
    cstep = step - 1
    k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))
    G = len(W_list)
    E = DimImg
    Bt = sum(ST[cstep]) #発話文中の単語数
    
    if (step == 1):  #初期設定
      ct = 1
      it = 1
      ccitems = [(1,1)]
      icitems = [(1,1)]
      
      phi = [np.array([1])]
      pi = np.array([1])
      
      #Cの場所概念にStのカウントを足す
      Nlg_c = [ sum( [np.array(ST[0])] ) ]
      Wc_temp = [(np.array(Nlg_c[0]) + beta0 ) / (sum(Nlg_c[0]) + G*beta0)]
      W = [Wc_temp[0]]
      
      #Cの場所概念にFtのカウントを足す
      Nle_c = [ sum( [np.array(FT[0])] ) ]
      thetac_temp = [(np.array(Nle_c[0]) + chi0 ) / (sum(Nle_c[0]) + E*chi0)]
      theta = [thetac_temp[0]]
      
      nk = 1
      xk = [np.array([XT[0].x, XT[0].y])]
      m_ML = sum(xk) #/ float(nk) #fsumではダメ
      kN = k0 + nk
      mN = ( k0*m0 + nk*m_ML ) / kN  #dim 次元横ベクトル
      nN = n0 + nk
      VN = V0 + sum([ np.dot( np.array([xk[0]]).T,np.array([xk[0]]) ) ]) + k0m0m0 - kN*np.dot(np.array([mN]).T,np.array([mN])) 
      MU = [mN]
      SIG = [np.array(VN) / (nN - dimx - 1)]
      weight_log = 0.0 #XT[cstep].weight
      
    else:
      #CTとITのインデックスは0から順番に整理されているものとする->しない
      cc = collections.Counter(CT) #｛Ct番号：カウント数｝
      L = len(cc)   #場所概念の数
      
      ic = collections.Counter(IT) #｛it番号：カウント数｝
      K = len(ic)   #位置分布の数
      
      ccitems = cc.items()  # [(ct番号,カウント数),(),...]
      cclist = [ccitems[c][1] for c in xrange(L)] #各場所概念ｃごとのカウント数
      
      icitems = ic.items()  # [(it番号,カウント数),(),...]
      iclist = [icitems[i][0] for i in xrange(K)] #各位置分布iごとのindex番号
      
      #場所概念のindexごとにITを分解
      ITc = [[] for c in xrange(L)]
      c = 0
      for s in xrange(cstep):
        for c in xrange(L):
          if (ccitems[c][0] == CT[s]):  ##場所概念のインデックス集合ccに含まれるカテゴリ番号のみ
            #print s,c,CT[s],ITc[c]
            ITc[c].append(IT[s])
            c += 1
      icc = [collections.Counter(ITc[c]) for c in xrange(L)] #｛cごとのit番号：カウント数｝の配列
      #Kc = [len(icc[c]) for c in xrange(L)]  #場所概念ｃごとの位置分布の種類数≠カウント数
      
      icclist = [ [icc[c][iclist[i]] for i in xrange(K)] for c in xrange(L)] #場所概念ｃにおける各位置分布iごとのカウント数
      #print np.array(icclist)
      
      Nct   = sum(cclist) #場所概念の総カウント数=データ数
      #Nit_l = sum(np.array(iccList))  #場所概念ｃごとの位置分布の総カウント数＝ 場所概念ｃごとのカウント数
      #これはcclistと一致するはず
      print Nct,cclist #, Nit_l
      
      CRP_CT  = np.array(cclist + [alpha0]) / (Nct + alpha0)
      
      icclist2 = [[icclist[c][i] for i in xrange(K)] for c in xrange(L)]
      for c in xrange(L):
        for i in xrange(K):
          if (icclist2[c][i] == 0):
            icclist2[c][i] = gamma0
      CRP_ITC = np.array( [np.array([icclist2[c][i] for i in xrange(K)] + [gamma0]) / (cclist[c] + gamma0) for c in xrange(L)] + [np.array([0.0 for i in xrange(K)] + [1.0])] )
      
      
      
      #print Xp
      xt = np.array([XT[cstep].x, XT[cstep].y])
      tpdf = [1.0 for i in xrange(K+1)]
      
      for k in xrange(K+1):
        #データがあるかないか関係のない計算
        #事後t分布用のパラメータ計算
        if (k == K):  #k is newの場合
          nk = 0;
        else:
          nk = ic[icitems[k][0]]  #icitems[k][1]
        ###kについて、zaが同じものを集める
        if nk != 0 :  #もしzaの中にkがあれば(計算短縮処理)        ##0ワリ回避
            xk = []
            for s in xrange(step-1) : 
              if IT[s] == icitems[k][0] : 
                xk = xk + [ np.array([XT[s].x, XT[s].y]) ]
            m_ML = sum(xk) / float(nk) #fsumではダメ
            print "K%d n:%d m_ML:%s" % (k,nk,str(m_ML))
            
            ##ハイパーパラメータ更新
            kN = k0 + nk
            mN = ( k0*m0 + nk*m_ML ) / kN  #dim 次元横ベクトル
            nN = n0 + nk
            #VN = V0 + sum([np.dot(np.array([xk[j]-m_ML]).T,np.array([xk[j]-m_ML])) for j in xrange(nk)]) + (k0*nk/kN)*np.dot(np.array([m_ML-m0]).T,np.array([m_ML-m0])) #旧バージョン
            VN = V0 + sum([np.dot(np.array([xk[j]]).T,np.array([xk[j]])) for j in xrange(nk)]) + k0m0m0 - kN*np.dot(np.array([mN]).T,np.array([mN]))  #speed up? #NIWを仮定した場合、V0は逆行列にしなくてよい
            if (VN[0][0] <= 0 or VN[1][1] <= 0 ):
              print "ERROR!!!! Posterior parameter VN is negative."
              print VN
            
        else:  #データがないとき
            kN = k0
            mN = m0
            nN = n0
            VN = V0
            
        #t分布の事後パラメータ計算
        mk = mN
        dofk = nN - dimx + 1
        InvSigk = (VN * (kN +1)) / (kN * dofk)
        
        #ｔ分布の計算
        #logt = log_multivariate_t_distribution( xt, mu, Sigma, dof)  ## t(x|mu, Sigma, dof)
        tpdf[k] = multivariate_t_distribution( xt, mk, InvSigk, dofk )  ## t(x|mu, Sigma, dof)
        #print "tpdf",k,tpdf[k][0]
        
      
      #ctとitの組をずらっと横に並べる（ベクトル）->2次元配列で表現 (temp[L+1][K+1]) [L=new][K=exist]は0
      temp2 = np.array([[10.0**10 * tpdf[i][0] * CRP_ITC[c][i] * CRP_CT[c] for i in xrange(K+1)] for c in xrange(L+1)])
      
      St_prob = [1.0 for c in xrange(L+1)]
      Ft_prob = [1.0 for c in xrange(L+1)]
      
      #位置分布kの計算と場所概念cの計算を分ける(重複計算を防ぐ)
      for l in xrange(L+1):
          if (l < L):
            ##単語のカウント数の計算
            #STはすでにBOWなのでデータstepごとのカウント数になっている
            Nlg = sum([np.array(ST[s])*(CT[s]==ccitems[l][0]) for s in xrange(step-1)])  #sumだとできる
            W_temp_log = np.log(np.array(Nlg) + beta0 ) - log(sum(Nlg) + G*beta0)
            St_prob[l] = exp(sum(np.array(W_temp_log) * np.array(ST[cstep])))
            
            ##画像特徴のカウント数の計算
            Nle = sum([np.array(FT[s])*(CT[s]==ccitems[l][0]) for s in xrange(step-1)])  #sumだとできる
            theta_temp_log = np.log(np.array(Nle) + chi0 ) - log(sum(Nle) + E*chi0)
            Ft_prob[l] = exp(sum(np.array(theta_temp_log) * np.array(FT[cstep]))) #.prod() #要素積
          else:  #ct=lかつit=kのデータがない場合
            St_prob[l] = 1.0/(G**Bt)
            Ft_prob[l] = 1.0/E  ##画像特徴は全次元足して１になるのでこれで良い
          
          temp2[l] = temp2[l] * St_prob[l] * Ft_prob[l]
      
      print temp2
      
      #2次元配列を1次元配列にする
      c,i = 0,0
      temp = []
      cxi_index_list = []
      for v in temp2:
        i = 0
        for item in v:
          temp.append(item)
          cxi_index_list.append((c,i))   #一次元配列に2次元配列のindexを対応付け
          i += 1
        c += 1
      
      temp = np.array(temp) / np.sum(temp)  #正規化
      #print temp
      cxi_index = list(multinomial(1,temp)).index(1)
      
      #1次元配列のindexを2次元配列に戻す（Ctとitに分割する）
      C,I = cxi_index_list[cxi_index]
      #print C,I
      
      Kp = K
      Lp = L
      #ct,itがNEWクラスターなら新しい番号を与える
      if C == L:
        ct = max(cc)+1
        print "c="+ str(ct) +" is new."
        Lp += 1
      else:
        ct = ccitems[C][0]
      if I == K:
        it = max(ic)+1
        print "i="+ str(it) +" is new."
        Kp += 1
      else:
        it = icitems[I][0]
      
      #print ct,it
      print "C", C, ", ct", ct, "; I", I, ", it", it
      
      #重みの計算
      wz_log = XT[cstep].weight
      
      #P(Ft|F{1:t},c{1:t-1},α,χ)の計算
      wf = sum( [Ft_prob[c] * CRP_CT[c] for c in xrange(L+1)])
      wf_log = log(wf)
      #print wf, wf_log
      
      #P(St|S{1:t},c{1:t-1},α,β)の計算
      psc = sum( [St_prob[c] * CRP_CT[c] for c in xrange(L+1)])
      if (UseLM == 1):
        #単なる単語の生起確率（頻度+βによるスムージング）:P(St|S{1:t-1},β)
        Ng = sum([np.array(ST[s]) for s in xrange(step-1)])  #sumだとできる
        W_temp2_log = np.log(np.array(Ng) + beta0 ) - log(sum(Ng) + G*beta0)
        ps_log = sum(np.array(W_temp2_log) * np.array(ST[cstep]))
        
        ws_log = log(psc) - ps_log
      else:
        ws_log = log(psc)
      #print log(psc), ps_log, ws_log
      
      #weight_log = wz_log+wf_log+ws_log   #sum of log probability
      weight_log = wf_log+ws_log   #sum of log probability
      print wz_log,wf_log,ws_log 
      print weight_log, exp(weight_log)
      
      #場所概念パラメータΘの更新処理
      #C,Kのカウントを増やす
      if C == L:  #L is new
        cclist.append(1)
        ccitems.append((ct,1))  ##ccitemsも更新する
        if I == K:  #K is new
          for c in xrange(L):
            icclist[c].append(0)  #既存の各場所概念ごとに新たな位置分布indexを増やす
        icclist.append([1*(k==I) for k in xrange(Kp)])  #新たな場所概念かつ新たな位置分布のカウント
      else:  #L is exist
        cclist[C] += 1
        if I == K:  #K is new
          for c in xrange(L):
            icclist[c].append(0)  #既存の各場所概念ごとに新たな位置分布indexを増やす
        icclist[C][I] += 1
      
      #pi = (np.array(cclist) + alpha0) / (Nct+1 + alpha0*Lp)  #np.array(iccList) /  (Nct + alpha0)
      #phi = [(np.array(icclist[c]) + gamma0) / (cclist[c] + gamma0*Kp) for c in xrange(Lp)]
      pi = (np.array(cclist) + alpha0/float(Lp)) / (Nct+1 + alpha0)  #np.array(iccList) /  (Nct + alpha0)
      phi = [(np.array(icclist[c]) + gamma0/float(Kp)) / (cclist[c] + gamma0) for c in xrange(Lp)]
      
      CT.append( ct )
      IT.append( it )
      #Cの場所概念にStのカウントを足す
      Nlg_c = [sum([np.array(ST[s])*(CT[s]==ccitems[c][0]) for s in xrange(step)]) for c in xrange(Lp)] 
      Wc_temp = [(np.array(Nlg_c[c]) + beta0 ) / (sum(Nlg_c[c]) + G*beta0) for c in xrange(Lp)]
      W = [Wc_temp[c] for c in xrange(Lp)]
      
      #Cの場所概念にFtのカウントを足す
      Nle_c = [sum([np.array(FT[s])*(CT[s]==ccitems[c][0]) for s in xrange(step)]) for c in xrange(Lp)] 
      thetac_temp = [(np.array(Nle_c[c]) + chi0 ) / (sum(Nle_c[c]) + E*chi0) for c in xrange(Lp)]
      theta = [thetac_temp[c] for c in xrange(Lp)]
      
      #Iの位置分布にxtのカウントを足す
      mNp = [[] for k in xrange(Kp)]
      nNp = [[] for k in xrange(Kp)]
      VNp = [[] for k in xrange(Kp)]
      
      for k in xrange(K):
        nk = ic[icitems[k][0]]
        if(k == I):  ##既存クラスのとき
          nk = nk + 1
          #print icitems[k][0], ic[icitems[k][0]]
        print k,nk
        if (nk != 0):  #0にはならないはずだが一応
          xk= []
          for s in xrange(step):
            if IT[s] == icitems[k][0]:
              xk = xk + [ np.array([XT[s].x, XT[s].y]) ]
          #IT_step = I #本来はI==kのカテゴリだけ値を更新すればよい
          #if (IT_step == k):
          #    xk = xk + [ np.array([XT[cstep].x, XT[cstep].y]) ]
          
          m_ML = sum(xk) / float(nk) #fsumではダメ
          ##ハイパーパラメータ更新
          kN = k0 + nk
          mN = ( k0*m0 + nk*m_ML ) / kN  #dim 次元横ベクトル
          nN = n0 + nk
          VN = V0 + sum([np.dot(np.array([xk[j]]).T,np.array([xk[j]])) for j in xrange(nk)]) + k0m0m0 - kN*np.dot(np.array([mN]).T,np.array([mN]))  #speed up? #NIWを仮定した場合、V0は逆行列にしなくてよい
          if (VN[0][0] <= 0 or VN[1][1] <= 0 ):
              print "ERROR!!!! Posterior parameter VN is negative.",k
              print VN
              print V0
              print sum([np.dot(np.array([xk[j]]).T,np.array([xk[j]])) for j in xrange(nk)])
              print k0m0m0
              print kN*np.dot(np.array([mN]).T,np.array([mN]))
              VN[0][0] = V0[0][0]
              VN[1][1] = V0[1][1]
        else:
          print "Error. nk["+str(k)+"]="+str(nk)
          kN = k0
          mN = m0
          nN = n0
          VN = V0
        
        mNp[k] = mN
        nNp[k] = nN
        VNp[k] = VN
      
      if (I == K): #新規クラスがあるとき
        nk = 1
        print I,nk
        xk = [np.array([XT[cstep].x, XT[cstep].y])]
        m_ML = sum(xk) / float(nk) #fsumではダメ
        kN = k0 + nk
        mN = ( k0*m0 + nk*m_ML ) / kN  #dim 次元横ベクトル
        nN = n0 + nk
        VN = V0 + sum([np.dot(np.array([xk[j]]).T,np.array([xk[j]])) for j in xrange(nk)]) + k0m0m0 - kN*np.dot(np.array([mN]).T,np.array([mN])) 
        if (VN[0][0] <= 0 or VN[1][1] <= 0 ):
              print "ERROR!!!! Posterior parameter VN is negative.",K
              print VN
              VN[0][0] = V0[0][0]
              VN[1][1] = V0[1][1]
        
        mNp[K] = mN
        nNp[K] = nN
        VNp[K] = VN
      
      MU = [mNp[k] for k in xrange(Kp)]
      SIG = [np.array(VNp[k]) / (nNp[k] - dimx - 1) for k in xrange(Kp)]
      
      
    ######################################################################
    ####                       ↑学習フェーズ↑                       ####
    ######################################################################
    loop = 1
    ########  ↓ファイル出力フェーズ↓  ########
    if loop == 1:
        #最終学習結果を出力
        #print "--------------------"
        print u"- <COMPLETED> Learning of Spatial Concepts in Particle:" + str(particle) + " -"
        #print "--------------------"
        
        #ファイルに保存
        SaveParameters(filename, particle, phi, pi, W, theta, MU, SIG)
        ###実際のindex番号と処理上のindex番号の対応付けを保存（場所概念パラメータΘは処理上の順番）
        WriteIndexData(filename, particle, ccitems, icitems,ct,it)
        
    ########  ↑ファイル出力フェーズ↑  ########
    return ct, it, weight_log


########################################
if __name__ == '__main__':
    import sys
    import os.path
    #import rospy
    #from std_msgs.msg import String
    from __init__ import *
    from JuliusLattice_gmm_LM import *
    import time
    import random
    
    #pub = rospy.Publisher('chatter', String, queue_size=10)
    #rospy.init_node('learn_SpCoSLAM')
    #clocktime = float(rospy.get_time()) ##rosbagがpause中のためとってこれない
    
    
    #trialname は上位プログラム（シェルスクリプト）から送られる
    #上位プログラムがファイル作成も行う（最初だけ）
    trialname = sys.argv[1]
    print trialname
    
    datasetNUM = sys.argv[2]
    datasetname = datasets[int(datasetNUM)]
    print datasetname #datasetPATH
    #print trialname
    
    Xp,step,m_count,CT,IT = ParticleSearcher(trialname)
    #step = 4
    print "step", step, "m_count", m_count
    
    teachingtime = []
    for line in open( datasetfolder + datasetname + 'teaching.csv', 'r'):
      #itemList = line[:].split(',')
      teachingtime.append(float(line))
    
    clocktime = float(teachingtime[step-1]) ##
    
    #出力ファイル名を要求
    #filename = raw_input("trialname?(folder) >")
    filename = datafolder + trialname + "/" + str(step)  ##FullPath of learning trial folder
    Makedir( filename ) #シェルスクリプトとかで先に作る?
    
    
    
    
    p_weight_log = np.array([0.0 for i in xrange(R)])
    p_weight = [0.0 for i in xrange(R)]
    W_list   = [[] for i in xrange(R)]
    
    if (UseFT == 1):
      FT = ReadImageData2(trialname, datasetname, step, clocktime)
    else:
      FT = [[0 for e in xrange(DimImg)] for s in xrange(step)]
    
    #パーティクルごとに計算
    for i in xrange(R):
      print "--------------------------------------------------"
      print "Particle:",i
      Julius_lattice(step,filename,trialname,i)    ##音声認識、ラティス形式出力、opemFST形式へ変換###########
      while (os.path.exists(filename + "/fst_gmm/" +str(i)+ "/"+ str(step-1).zfill(3) +".fst" ) != True):
        print filename + "/fst_gmm/"  +str(i)+ "/"+ str(step).zfill(3) + ".fst", "wait(30s)... or ERROR?"
        time.sleep(1.0) #sleep(秒指定)
        Julius_lattice(step,filename,trialname,i)
      print "Julius complete!"
      
      if (ramdoman != 0):
        annealsteps  += random.randint(-1*annealsteps+1,ramdoman)
        anneallength += random.randint(-1*anneallength+1,ramdoman)

      #print "latticelm run. sample_num:" + str(sample)
      latticelm_CMD = "latticelm -input fst -filelist "+ filename + "/fst_gmm/" +str(i)+ "/fstlist.txt -prefix " + filename + "/out_gmm/" +str(i)+ "/" + str(i) + "_ -symbolfile " + filename + "/fst_gmm/" +str(i)+ "/isyms.txt -burnin " + str(burnin) + " -samps " + str(samps) + " -samprate " + str(samprate) + " -knownn " + str(knownn) + " -unkn " + str(unkn) + " -annealsteps " + str(annealsteps) + " -anneallength " + str(anneallength)
      
      p = os.popen( latticelm_CMD )   ##latticelm ###################
      #time.sleep(1.0) #sleep(秒指定)###################
      p.close()###########################
      latticelm_count = 0
      while (os.path.exists(filename + "/out_gmm/" +str(i)+ "/" + str(i) + "_samp."+str(samps) ) != True):
              print filename + "/out_gmm/" +str(i)+ "/" + str(i) + "_samp."+str(samps),"wait(10s)... or ERROR?"
              #p.close()
              latticelm_count += 1
              if (latticelm_count > 10):
                print "run latticelm again."
                p = os.popen( latticelm_CMD )   ##latticelm 
                p.close()
                latticelm_count = 0
              time.sleep(1.0) #sleep(秒指定)
      #p.close()###########################
      print "Particle:",i," latticelm complete!"
      
      
      #for i in xrange(R): ###############
      print "--------------------------------------------------" ###############
      print "Particle:",i ###############
      
      W_list[i], ST = ReadWordData(step, filename, i)
      print "Read word data."
      #CT, IT = ReaditCtData(trialname, step, i)
      print "Read Ct,it data."
      print "CT",CT[i]
      print "IT",IT[i]
      ct, it, p_weight_log[i] = Learning(step, filename, i, Xp[i], ST, W_list[i], CT[i], IT[i], FT)     ##場所概念の学習
      print "Particle:",i," Learning complete!"
      
      WriteParticleData(filename, step, i, Xp[i], p_weight_log[i], ct, it, CT[i], IT[i])  #重みは正規化されてない値が入る
      WriteWordData(filename, i, W_list[i])
      
      print "Write particle data and word data."
      
      WordDictionaryUpdate(step, filename, W_list[i], i)       ##単語辞書登録
      print "Language Model update!"
    
    print "--------------------------------------------------" ###############
    #logの最大値を引く処理
    print p_weight_log
    logmax = max(p_weight_log)
    p_weight_log = p_weight_log - logmax  #np.arrayのため
    
    WriteWeightData(trialname, m_count, p_weight_log)
    
    #print p_weight_log
    #weightの正規化
    p_weight = np.exp(p_weight_log)
    sum_weight = sum(p_weight)
    p_weight = p_weight / sum_weight
    print "Weight:",p_weight
    
    MAX_weight_particle = np.argmax(p_weight) #p_weight.index(max(p_weight))                     ##最大重みのパーティクルindex
    print MAX_weight_particle
    
    
    ##最大重みのパーティクルindexと重みを保存する
    fp = open(filename + "/weights.csv", 'w')
    fp.write(str(MAX_weight_particle) + '\n')
    for particle in xrange(R):
      fp.write(str(p_weight[particle]) + ',')
    fp.write('\n')
    fp.close()
    
    #処理終了のフラグを送る
    #endflag = "1"
    #pub.publish(endflag)
    
    flag = 0
    fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
    fp.write(str(flag))
    fp.close()
    
    fp = open( datafolder + trialname + "/gwaitflag.txt", 'w')
    fp.write(str(m_count+1))
    fp.close()
    
########################################

