#coding:utf-8
#PRR評価用プログラム（範囲はデータセットから決まる版）
#Akira Taniguchi (2017/02/27)
import sys
import os.path
import random
import string
import collections
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from __init__ import *
import os.path
from Julius1best_gmm import *
#import time

##追加の評価指標のプログラム
##発話→位置の評価：p(xt|Ot)
step = 50


#相互推定のプログラムにimportして使う。
#プログラムが単体でも動くようにつくる。

#各関数、編集途中。

def gaussian(x,myu,sig):
    ###1次元ガウス分布
    gauss = (1.0 / sqrt(2.0*PI*sig*sig)) * exp(-1.0*(float((x-myu)*(x-myu))/(2.0*sig*sig)))
    return gauss
    
def gaussian2d(Xx,Xy,myux,myuy,sigma):
    ###ガウス分布(2次元)
    sqrt_inb = float(1) / ( 2.0 * PI * sqrt( np.linalg.det(sigma)) )
    xy_myu = np.array( [ [float(Xx - myux)],[float(Xy - myuy)] ] )
    dist = np.dot(np.transpose(xy_myu),np.linalg.solve(sigma,xy_myu))
    gauss2d = (sqrt_inb) * exp( float(-1/2) * dist )
    return gauss2d

def fill_param(param, default):   ##パラメータをNone の場合のみデフォルト値に差し替える関数
    if (param == None): return default
    else: return param

def invwishartrand_prec(nu,W):
    return inv(wishartrand(nu,W))

def invwishartrand(nu, W):
    return inv(wishartrand(nu, inv(W)))

def wishartrand(nu, W):
    dim = W.shape[0]
    chol = cholesky(W)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.axrange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in xrange(dim):
        for j in xrange(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = np.random.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))
    
class NormalInverseWishartDistribution(object):
#http://stats.stackexchange.com/questions/78177/posterior-covariance-of-normal-inverse-wishart-not-converging-properly
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = float(lmbda)
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)

    def r(self):
        sigma = np.linalg.inv(self.wishartrand())
        return (np.random.multivariate_normal(self.mu, sigma / self.lmbda), sigma)

    def wishartrand(self):
        dim = self.inv_psi.shape[0]
        chol = np.linalg.cholesky(self.inv_psi)
        foo = np.zeros((dim,dim))
        
        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    foo[i,j] = np.sqrt(chi2.rvs(self.nu-(i+1)+1))
                else:
                    foo[i,j]  = np.random.normal(0,1)
        return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

    def posterior(self, data):
        n = len(data)
        mean_data = np.mean(data, axis=0)
        sum_squares = np.sum([np.array(np.matrix(x - mean_data).T * np.matrix(x - mean_data)) for x in data], axis=0)
        mu_n = (self.lmbda * self.mu + n * mean_data) / (self.lmbda + n)
        lmbda_n = self.lmbda + n
        nu_n = self.nu + n
        psi_n = self.psi + sum_squares + self.lmbda * n / float(self.lmbda + n) * np.array(np.matrix(mean_data - self.mu).T * np.matrix(mean_data - self.mu))
        return NormalInverseWishartDistribution(mu_n, lmbda_n, nu_n, psi_n)

def levenshtein_distance(a, b):
    m = [ [0] * (len(b) + 1) for i in range(len(a) + 1) ]

    for i in xrange(len(a) + 1):
        m[i][0] = i

    for j in xrange(len(b) + 1):
        m[0][j] = j

    for i in xrange(1, len(a) + 1):
        for j in xrange(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                x = 0
            else:
                x = 1
            m[i][j] = min(m[i - 1][j] + 1, m[i][ j - 1] + 1, m[i - 1][j - 1] + x)
    # print m
    return m[-1][-1]
    

#http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section5_2-Dirichlet-Processes.ipynb
def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()

#http://stackoverflow.com/questions/13903922/multinomial-pmf-in-python-scipy-numpy
class Multinomial(object):
  def __init__(self, params):
    self._params = params

  def pmf(self, counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 1.
    for i,c in enumerate(counts):
      prob *= self._params[i]**counts[i]

    return prob * exp(self._log_multinomial_coeff(counts))

  def log_pmf(self,counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 0.
    for i,c in enumerate(counts):
      prob += counts[i]*log(self._params[i])

    return prob + self._log_multinomial_coeff(counts)

  def _log_multinomial_coeff(self, counts):
    return self._log_factorial(sum(counts)) - sum(self._log_factorial(c)
                                                    for c in counts)

  def _log_factorial(self, num):
    if not round(num)==num and num > 0:
      raise ValueError("Can only compute the factorial of positive ints")
    return sum(log(n) for n in range(1,num+1))


#itとCtのデータを読み込む（教示した時刻のみ）
def ReaditCtData(trialname, cstep, particle):
  CT,IT = [0 for i in xrange(step)],[0 for i in xrange(step)]
  i = 0
  if (step != 0):  #最初のステップ以外
    for line in open( datafolder + trialname + "/" + str(cstep) + "/particle" + str(particle) + ".csv" , 'r' ):
      itemList = line[:-1].split(',')
      CT[i] = int(itemList[7]) 
      IT[i] = int(itemList[8]) 
      i += 1
  return CT, IT

# Reading particle data (ID,x,y,theta,weight,previousID)
def ReadParticleData2(step, particle, trialname):
  p = []
  for line in open ( datafolder + trialname + "/"+ str(step) + "/particle" + str(particle) + ".csv" ):
    itemList = line[:-1].split(',')
    p.append( [float(itemList[2]), float(itemList[3])] )
    #p.append( Particle( int(itemList[0]), float(itemList[1]), float(itemList[2]), float(itemList[3]), float(itemList[4]), int(itemList[5])) )
  return p



###↓###発話→場所の認識############################################
def Location_from_speech(cstep, trialname, THETA, particle, L,K):
  datasetNUM = 0
  datasetname = datasets[int(datasetNUM)]
  
  #教示位置データを読み込み平均値を算出（xx,xy）
  XX = []
  count = 0
  
  ItC = []
  #それぞれの場所の中央座標を出す（10カ所）
  s = 0
  #正解データを読み込みIT
  for line in open(datasetfolder + datasetname + correct_It, 'r'):
      itemList = line[:].split(',')
      for i in xrange(len(itemList)):
        if (itemList[i] != '') and (s < step):
          ItC = ItC + [int(itemList[i])]
        s += 1
        
  ic = collections.Counter(ItC)
  icitems = ic.items()  # [(it番号,カウント数),(),...]
  
  #if (data_name != 'test000'):
  if (1):
        #Xt = []
        #最終時点step=50での位置座標を読み込み(これだと今のパーティクル番号の最終時刻の位置情報になるが細かいことは気にしない)
        Xt = np.array( ReadParticleData2(step,particle, trialname) )
  
  X = [[] for i in range(len(ic))]
  Y = [[] for i in range(len(ic))]
  for j in xrange(len(ic)):  #教示場所の種類数
          Xtemp  = []
          for i in xrange(len(ItC)): #要はステップ数（=50）
            if (icitems[j][0] == ItC[i]):
              Xtemp = Xtemp + [Xt[i]]
              X[j] = X[j] + [Xt[i][0]]
              Y[j] = Y[j] + [Xt[i][1]]
          
          #print len(Xtemp),Xtemp,ic[icitems[j][0]]
          #XX = XX + [sum(np.array(Xtemp))/float(ic[icitems[j][0]])]
          #XY[j] = Xtemp
          
  """
  ###SIGVerse###
  HTW = []
  for line2 in open('./../r/' + data_name +  '_HTW.csv', 'r'):
    itemList2 = line2[:-1].split(',')
    HTW = HTW + [itemList2[0]]
    
  #i = 0
  Xt_temp = []
  Xt = [[0.0,0.0] for n in xrange(len(HTW)) ]
  #TN = []
  for line3 in open('./../r/' + data_name +  '_X_true.csv', 'r'):
    itemList3 = line3[:-1].split(',')
    Xt_temp = Xt_temp + [[float(itemList3[2]) + 500, float(itemList3[1]) + 250]]
    #TN = TN + [i]
    #print TN
    #i = i + 1
  
  #げんかん(ge)0-9、てえぶるのあたり黒(teb)10-19、白(tew)20-29、ほんだな(hd)30-39、
  #そふぁあまえ(sf)40-49、きっちん(kt)50-59、だいどころ(dd)60-69、ごみばこ(go)70-79、てれびまえ(tv)80-89
  
  #てえぶるのあたりはどちらかにいけば正解、だいどころときっちんはどちらの発話でも同じ領域に行けば正解にする処理が必要
  
  ge = 0
  teb = 10
  tew = 20
  hd = 30
  sf = 40
  kt = 50
  dd = 60
  go = 70
  tv = 80
  
  for i in xrange(len(HTW)):
    htw = HTW[i]
    if (htw == "ge"):
      Xt[ge] = Xt_temp[i]
      ge = ge + 1
    
    if (htw == "teb"):
      Xt[teb] = Xt_temp[i]
      teb = teb + 1
    
    if (htw == "tew"):
      Xt[tew] = Xt_temp[i]
      tew = tew + 1
    
    if (htw == "hd"):
      Xt[hd] = Xt_temp[i]
      hd = hd + 1
      
    if (htw == "sf"):
      Xt[sf] = Xt_temp[i]
      sf = sf + 1
      
    if (htw == "kt"):
      Xt[kt] = Xt_temp[i]
      kt = kt + 1
      
    if (htw == "dd"):
      Xt[dd] = Xt_temp[i]
      dd = dd + 1
      
    if (htw == "tv"):
      Xt[tv] = Xt_temp[i]
      tv = tv + 1
      
    if (htw == "go"):
      Xt[go] = Xt_temp[i]
      go = go + 1
    
    X = [Xt[i][0] for i in range(len(HTW))]
    Y = [Xt[i][1] for i in range(len(HTW))]
    
  """
  print X
  print Y
  
  #THETA = [W,W_index,Myu,S,pi,phi_l]
  W = THETA[0]
  W_index = THETA[1]
  Myu = THETA[2]
  S = THETA[3]
  pi = THETA[4]
  phi_l = THETA[5]
  theta = THETA[6]
  
  
  
  ##自己位置推定用の音声ファイルを読み込み
  # wavファイルを指定
  files = glob.glob(speech_folder_go)   #./../../../Julius/directory/CC3Th2/ (相対パス)
  #genkan,teeburu,teeburu,hondana,sofa,kittin,daidokoro,gomibako,terebimae
  files.sort()
  
  LAR = [] #0.0
  
  ##パーティクルをばらまく（全ての各位置分布に従う点をサンプリング）
  Xp = []
  
  for j in range(K):
    #x1,y1 = np.random.multivariate_normal([Myu[j][0][0],Myu[j][1][0]],S[j],1).T
    #位置分布の平均値と位置分布からサンプリングした99点の１位置分布に対して合計100点をxtの候補とした
    for i in range(9):    
      x1,y1 = np.mean(np.array([ np.random.multivariate_normal([Myu[j][0][0],Myu[j][1][0]],S[j],1).T ]),0)
      Xp = Xp + [[x1,y1]]
      print x1,y1
    Xp = Xp + [[Myu[j][0][0],Myu[j][1][0]]]
    print Myu[j][0][0],Myu[j][1][0]
    
  filename = datafolder + trialname + "/" + str(cstep)  ##FullPath of learning trial folder
  if ("nl" in trialname) or ("p1" in trialname):
      UseLM = 1
      WordDictionaryUpdate2(cstep, filename, W_index)       ##単語辞書登録
  else:
      UseLM = 1  
      WordDictionaryUpdate2(cstep, filename, W_index)       ##単語辞書登録
      
  k = 0
  ##学習した単語辞書を用いて音声認識し、BoWを得る
  for f in files:
    St = RecogLattice( f , cstep , filename, trialname , N_best_number)
    #print St
    Otb_B = [0 for i in xrange(len(W_index))] #[[] for j in range(len(St))]
    for j in range(len(St)):
      for i in range(5):
              St[j] = St[j].replace(" <s> ", "")
              St[j] = St[j].replace("<sp>", "")
              St[j] = St[j].replace(" </s>", "")
              St[j] = St[j].replace("  ", " ") 
              St[j] = St[j].replace("\n", "") 
              
      print j,St[j]
      Otb = St[j].split(" ")
      ##データごとにBOW化
      #Otb_B = [ [] for s in xrange(len(files)) ]
      #for n in xrange(len(files)):
      #  Otb_B[n] = [0 for i in xrange(len(W_index))]
      #Otb_B = [0 for i in xrange(len(W_index))]
      
      #for n in xrange(N):
      for j2 in xrange(len(Otb)):
          #print n,j,len(Otb_Samp[r][n])
          for i in xrange(len(W_index)):
            #print W_index[i].decode('sjis'),Otb[j]
            if (W_index[i].decode('sjis') == Otb[j2] ):
            #####if (W_index[i].decode('utf8') == Otb[j] ):
              Otb_B[i] = Otb_B[i] + 1
              #print W_index[i].decode('sjis'),Otb[j]
    print particle,Otb_B
    
    
    
    pox = [0.0 for i in xrange(len(Xp))]
    ##パーティクルごとにP(xt|Ot,θ)の確率値を計算、最大の座標を保存
    ##位置データごとに
    for xdata in xrange(len(Xp)):
        
        ###提案手法による尤度計算####################
        #Ot_index = 0
        
        #for otb in xrange(len(W_index)):
        #Otb_B = [0 for j in xrange(len(W_index))]
        #Otb_B[Ot_index] = 1
        temp = [0.0 for c in range(L)]
        #print Otb_B
        for c in xrange(L) :
            ##場所の名前、多項分布の計算
            W_temp = Multinomial(W[c])
            temp[c] = W_temp.pmf(Otb_B)
            #temp[c] = W[c][otb]
            ##場所概念の多項分布、piの計算
            temp[c] = temp[c] * pi[c]
            
            ##itでサメーション
            it_sum = 0.0
            for it in xrange(K):
                if (S[it][0][0] < pow(10,-100)) or (S[it][1][1] < pow(10,-100)) :    ##共分散の値が0だとゼロワリになるので回避
                    if int(Xp[xdata][0]) == int(Myu[it][0]) and int(Xp[xdata][1]) == int(Myu[it][1]) :  ##他の方法の方が良いかも
                        g2 = 1.0
                        print "gauss 1"
                    else : 
                        g2 = 0.0
                        print "gauss 0"
                else : 
                    g2 = gaussian2d(Xp[xdata][0],Xp[xdata][1],Myu[it][0],Myu[it][1],S[it])  #2次元ガウス分布を計算
                it_sum = it_sum + g2 * phi_l[c][it]
                
            temp[c] = temp[c] * it_sum
        
        pox[xdata] = sum(temp)
        
        #print Ot_index,pox[Ot_index]
        #Ot_index = Ot_index + 1
        #POX = POX + [pox.index(max(pox))]
        
        #print pox.index(max(pox))
        #print W_index_p[pox.index(max(pox))]
        
    
    
    Xt_max = [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] #[0.0,0.0] ##確率最大の座標候補
    
    
    ##正解をどうするか
    ##正解の区間の座標であれば正解とする
    PXO = 0.0  ##座標が正解(1)か不正解か(0)
    
    #for i in range(K): #発話ごとに正解の場所の領域がわかるはず
    if (1):
      ##正解区間設定(上下左右10のマージン)margin
      #i = k
      print "k=",k
      
      if(k == 3): # ikidomari 2kasyo 
          #x座標の最小値-10
          xmin1 = min(X[4])
          #x座標の最大値+10
          xmax1 = max(X[4])
          #y座標の最小値-10
          ymin1 = min(Y[4])
          #y座標の最大値+10
          ymax1 = max(Y[4])
          
          #x座標の最小値-10
          xmin2 = min(X[5])
          #x座標の最大値+10
          xmax2 = max(X[5])
          #y座標の最小値-10
          ymin2 = min(Y[5])
          #y座標の最大値+10
          ymax2 = max(Y[5])
          
          #正解判定
          if( ((xmin1-margin <= Xt_max[0] <= xmax1+margin) and (ymin1-margin <= Xt_max[1] <= ymax1+margin)) or ((xmin2-margin <= Xt_max[0] <= xmax2+margin) and (ymin2-margin <= Xt_max[1] <= ymax2+margin)) ):
            PXO = PXO + 1
            print cstep,k,Xt_max," OK!"
          else:
            print cstep,k,Xt_max," NG!"
      
      elif(k == 1): # kyuukeijyo 2kasyo
          #x座標の最小値-10
          xmin1 = min(X[1])
          #x座標の最大値+10
          xmax1 = max(X[1])
          #y座標の最小値-10
          ymin1 = min(Y[1])
          #y座標の最大値+10
          ymax1 = max(Y[1])
          
          #x座標の最小値-10
          xmin2 = min(X[2])
          #x座標の最大値+10
          xmax2 = max(X[2])
          #y座標の最小値-10
          ymin2 = min(Y[2])
          #y座標の最大値+10
          ymax2 = max(Y[2])
          
          
          #正解判定
          if( ((xmin1-margin <= Xt_max[0] <= xmax1+margin) and (ymin1-margin <= Xt_max[1] <= ymax1+margin)) or ((xmin2-margin <= Xt_max[0] <= xmax2+margin) and (ymin2-margin <= Xt_max[1] <= ymax2+margin)) ):
            PXO = PXO + 1
            print cstep,k,Xt_max," OK!"
          else:
            print cstep,k,Xt_max," NG!"
      elif(k == 6 or k == 7): #purintaabeya and daidokoro
          #x座標の最小値-10
          xmin1 = min(X[8])
          #x座標の最大値+10
          xmax1 = max(X[8])
          #y座標の最小値-10
          ymin1 = min(Y[8])
          #y座標の最大値+10
          ymax1 = max(Y[8])
          
          #正解判定
          if( ((xmin1-margin <= Xt_max[0] <= xmax1+margin) and (ymin1-margin <= Xt_max[1] <= ymax1+margin)) ):
            PXO = PXO + 1
            print cstep,k,Xt_max," OK!"
          else:
            print cstep,k,Xt_max," NG!"
      
      else:
          if (k == 0):
            i = 0
          elif (k == 2):
            i = 3
          elif (k == 4):
            i = 6
          elif (k == 5):
            i = 7
          elif (k == 8):
            i = 9
          #x座標の最小値-10
          xmin = min(X[i]) #min(X[i*10:i*10 + 10])
          #x座標の最大値+10
          xmax = max(X[i])
          #y座標の最小値-10
          ymin = min(Y[i])
          #y座標の最大値+10
          ymax = max(Y[i])
          
          #正解判定
          if( (xmin-margin <= Xt_max[0] <= xmax+margin) and (ymin-margin <= Xt_max[1] <= ymax+margin) ):
            PXO = PXO + 1
            print cstep,k,Xt_max," OK!"
          else:
            print cstep,k,Xt_max," NG!"
      
    
    LAR = LAR + [PXO]
    k = k + 1
    
  
  #LARの平均値を算出(各発話ごとの正解の割合え)
  LAR_mean = sum(LAR) / float(len(LAR))
  print LAR
  print LAR_mean
  
  return LAR_mean
###↑###発話→場所の認識############################################


###↓###単語辞書読み込み書き込み追加############################################
#MAX_Samp : 重みが最大のパーティクル
def WordDictionaryUpdate2(step, filename, W_list):
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
  if (1):
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
  fp = open( filename + '/WDonly.htkdic', 'w')
  for list in xrange(len(LIST)):
    if (list < 3):
        fp.write(LIST[list])
  #if (UseLM == 1):
  if (1):
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

def Evaluation2(trialname):
  
  #相互推定の学習結果データを読み込む
  #MI_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  #ARI_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  #PARs_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  #PARw_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  PRR_List = [[0.0 for i in xrange(R)] for j in xrange(step)] 
  # location accuracy rate from a name of place 
  MAX_Samp = [0 for j in xrange(step)]
  
  L = [[0.0 for i in xrange(R)] for j in xrange(step)]
  K = [[0.0 for i in xrange(R)] for j in xrange(step)]
  
  #イテレーションごとに選ばれた学習結果の評価値をすべて保存するファイル
  fp = open(datafolder + trialname + '/' + trialname + '_EvaluationPRR.csv', 'w')  
  
  #相互推定のイテレーションと単語分割結果の候補のすべてのパターンの評価値を保存
  #fp_ARI = open('./data/' + filename + '/' + filename + '_A_sougo_ARI.csv', 'w')  
  #fp_PARs = open('./data/' + filename + '/' + filename + '_A_sougo_PARs.csv', 'w')  
  #fp_PARw = open('./data/' + filename + '/' + filename + '_A_sougo_PARw.csv', 'w')  
  #fp_MI = open('./data/' + filename + '/' + filename + '_A_sougo_MI.csv', 'w')  
  fp_PRR = open(datafolder + trialname + '/' + trialname + '_meanEvaluationPRR.csv', 'w')  
  #fp.write('MI,ARI,PARs,PARw\n')
  fp.write('PRR\n')
  
  #相互推定のイテレーションごとに
  for s in xrange(step):
    
    i = 0
    #重みファイルを読み込み
    for line in open(datafolder + trialname + '/'+ str(s+1) + '/weights.csv', 'r'):   ##読み込む
        #itemList = line[:-1].split(',')
        if (i == 0):
          MAX_Samp[s] = int(line)
        i += 1
    
    #最大尤度のパーティクル番号を保存
    particle = MAX_Samp[s]
    
    #各stepごとの全パーティクルの学習結果データを読み込む
    for r in xrange(R):
      #if (0):
      
      #各stepごとの最大尤度のパーティクル情報(CT,IT)を読み込む(for ARI)
      CT,IT = ReaditCtData(trialname, s+1, r)
      
      #推定されたLとKの数を読み込み
      i = 0
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/index' + str(r) + '.csv', 'r'):   ##読み込む
        itemList = line[:-1].split(',')
        #itemint = [int(itemList[j]) for j in xrange(len(itemList))]
        print itemList
        if (i == 0):
          #for item in itemList:
          L[s][r] = len(itemList) -1
        elif (i == 1):
          K[s][r] = len(itemList) -1
        i += 1
      
      W_index= []
      
      i = 0
      #テキストファイルを読み込み
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/W_list' + str(r) + '.csv', 'r'):   ##*_samp.100を順番に読み込む
        itemList = line[:-1].split(',')
        
        if(i == 0):
            for j in range(len(itemList)):
              if (itemList[j] != ""):
                W_index = W_index + [itemList[j]]
        i = i + 1
      
      #####パラメータW、μ、Σ、φ、πを入力する#####
      Myu = [ np.array([[ 0 ],[ 0 ]]) for i in xrange(K[s][r]) ]      #位置分布の平均(x,y)[K]
      S = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in xrange(K[s][r]) ]      #位置分布の共分散(2×2次元)[K]
      W = [ [0.0 for j in xrange(len(W_index))] for c in xrange(L[s][r]) ]  #場所の名前(多項分布：W_index次元)[L]
      theta = [ [0.0 for j in xrange(DimImg)] for c in xrange(L[s][r]) ] 
      pi = [ 0 for c in xrange(L[s][r])]     #場所概念のindexの多項分布(L次元)
      phi_l = [ [0 for i in xrange(K[s][r])] for c in xrange(L[s][r]) ]  #位置分布のindexの多項分布(K次元)[L]
      
      #Ct = []
      
      
      i = 0
      ##Myuの読み込み
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/mu' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #itemList[1] = itemList[1].replace("_"+str(particle), "")
        Myu[i] = np.array([[ float(itemList[0]) ],[ float(itemList[1]) ]])
        
        i = i + 1
      
      i = 0
      ##Sの読み込み
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/sig' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #itemList[2] = itemList[2].replace("_"+str(particle), "")
        S[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3]) ]])
        
        i = i + 1
      
      ##phiの読み込み
      c = 0
      #テキストファイルを読み込み
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/phi' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
            if itemList[i] != "":
              phi_l[c][i] = float(itemList[i])
        c = c + 1
      
      
      ##piの読み込み
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/pi' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            pi[i] = float(itemList[i])
      
      ##Wの読み込み
      c = 0
      #テキストファイルを読み込み
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/W' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              W[c][i] = float(itemList[i])
              
              #print itemList
        c = c + 1
      
      ##thetaの読み込み
      c = 0
      #テキストファイルを読み込み
      for line in open(datafolder + trialname + '/'+ str(s+1) + '/theta' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              theta[c][i] = float(itemList[i])
              
              #print itemList
        c = c + 1
      
      
      #############################################################
      
      print s,r
      
      #print "ARI"
      #ARI_List[s][r] = ARI(Ct)
      
      #print "PAR_S"
      #PARs_List[s][r] = PAR_sentence(s,r)
      
      pi_temp = [pi[i] for i in range(len(pi))]
      phi_l_temp = [ [phi_l[c][i] for i in xrange(K[s][r])] for c in xrange(L[s][r]) ]
      
      ni = [0 for i in range(len(pi))]
      for i in range(len(pi)):
        ni[i] = pi[i]*((s+1)+L[s][r]*alpha0)-alpha0
      
      #piのかさまし対策処理
      pi = [ ( ni[i]+(alpha0/float(L[s][r])) ) / float( (s+1)+alpha0 )  for i in range(len(pi))]
      #phiのかさまし対策処理
      phi_l = [ [( (phi_l[c][i]*(ni[c]+K[s][r]*gamma0)-gamma0)+(gamma0/float(K[s][r])) ) / float( ni[c]+gamma0 ) for i in xrange(K[s][r])] for c in xrange(L[s][r]) ]
      #phi = [ ( (phi[i]*((s+1)+L[s][r]*alpha)-alpha)+(alpha/float(L[s][r])) ) / float( (s+1)+alpha )  for i in range(len(pi))]
      
      i = 0
      for pi2 in pi:
        if(pi2 < 0.0):
          print pi2
          pi = pi_temp
        for phi2 in phi_l[i]:
          if (phi2 < 0.0):
            print phi2
            phi_l = phi_l_temp
        i = i + 1  
        
      THETA = [W,W_index,Myu,S,pi,phi_l,theta]
      #NOP = []
      #print "PAR_W"
      #PARw_List[s][r] = Name_of_Place(THETA)
      
      PRR_List[s][r] = Location_from_speech(s+1, trialname, THETA, r,  L[s][r], K[s][r])
      
      print "OK!"
      #fp_PRR.write(str( PRR_List[s][r] ))
      #fp_PRR.write(',')
    
    #fp_ARI.write(',')
    #smean = sum(ARI_List[s])/R
    #fp_ARI.write(str(smean))
    #fp_ARI.write('\n')
    
    #fp_PARs.write(',')
    #smean = sum(PARs_List[s])/R
    #fp_PARs.write(str(smean))
    #fp_PARs.write('\n')
    
    #fp_PARw.write(',')
    #smean = sum(PARw_List[s])/R
    #fp_PARw.write(str(smean))
    #fp_PARw.write('\n')
    
    #fp_PRR.write(',')
    smean = sum(PRR_List[s])/float(R)
    fp_PRR.write(str(smean))
    fp_PRR.write('\n')
    
    #MI,ARI,PARs,PARw,
    
    #fp.write( str(MI_List[s][MAX_Samp[s]])+','+ str(ARI_List[s][MAX_Samp[s]])+','+ str(PARs_List[s][MAX_Samp[s]])+','+str(PARw_List[s][MAX_Samp[s]]) )
    fp.write( str(PRR_List[s][MAX_Samp[s]]) )
    fp.write('\n')
    
  print "close."
  
  fp.close()
  #fp_ARI.close()
  #fp_PARs.close()
  #fp_PARw.close()
  #fp_MI.close()
  fp_PRR.close()
  
if __name__ == '__main__':
    #出力ファイル名を要求
    trialname = raw_input("trialname?(folder w/o number) >") #"tamd2_sig_mswp_01" 
    
    if ("p1" in trialname):
      R = 1
    elif ("p30" in trialname):
      R = 30
    
    if ("nf" in trialname):
      UseFT = 0
    else:
      UseFT = 1
    
    if ("nl" in trialname):
      UseLM = 0
    else:
      UseLM = 1
    """
    if ("sig" in filename):
      data_name = 'test000'
      L = 50
      K = 50
      kyouji_count = 90
      correct_Ct = 'Ct_correct.csv'  #データごとの正解のCt番号
      correct_data = 'TAMD1_human.txt'  #データごとの正解の文章（単語列、区切り文字つき）(./data/)
      correct_name = 'name_correct.csv'  #データごとの正解の場所の名前（音素列）
      
    else:
      data_name = 'datah.csv'
      L = 100
      K = 100
      kyouji_count = 100
      correct_Ct = 'Ct_correct_turtle.csv'  #データごとの正解のCt番号
      correct_data = 'TAMD1_human_turtle.csv'  #データごとの正解の文章（単語列、区切り文字つき）(./data/)
      correct_name = 'name_correct_turtle.csv'  #データごとの正解の場所の名前（音素列）
      
    if ("p" in filename):
      lang_init = 'phonemes.htkdic'
    else:
      lang_init = 'web.000.htkdic'
    if ("ms" in filename):
      step = 10
      R = 6
    if ("m0" in filename):
      step = 10
      R = 1
    if ("nakamura" in filename):
      step = 10
      R = 1
    if (("000t" in filename) or ("000b" in filename)):
      step = 1
      R = 1
    """
    for s in range(1,11): #１からstepの数字までのループ
      Evaluation2(trialname + str(s).zfill(3))
