#coding:utf-8
#PRR評価用プログラム（範囲指定版）
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
  
  
  #THETA = [W,W_index,Myu,S,pi,phi_l]
  W = THETA[0]
  W_index = THETA[1]
  Myu = THETA[2]
  S = THETA[3]
  pi = THETA[4]
  phi_l = THETA[5]
  theta = THETA[6]
  
  p_stlit = [[0.0 for w in range(len(W_index))] for i in range(K)]
  LAR = [] #0.0
  Otb_B = [[1*(i==j) for i in xrange(len(W_index))] for j in range(len(W_index))]
  
  
  ##学習した単語辞書を用いて音声認識し、BoWを得る
  for it in range(K):
    for otb in range(len(W_index)):
      temp_bunbo = sum([W[c2][otb]*pi[c2] for c2 in range(L)])
      temp = 0
      for c in range(L):
        temp += (W[c][otb] / temp_bunbo) * phi_l[c][it] * pi[c]
        
      p_stlit[it][otb] = temp
    
  
  #LARの平均値を算出(各発話ごとの正解の割合え)
  #LAR_mean = sum(LAR) / float(len(LAR))
  print p_stlit
  #print LAR_mean
  
  return p_stlit
###↑###発話→場所の認識############################################

# Reading particle data (ID,x,y,theta,weight,previousID)
def ReadParticleData3(step, particle, trialname):
  p = []
  pid = []
  for line in open ( datafolder + trialname + "/"+ str(step) + "/particle" + str(particle) + ".csv" ):
    itemList = line[:-1].split(',')
    p.append( [float(itemList[2]), float(itemList[3])] )
    pid.append( int(itemList[1]) )
    #p.append( Particle( int(itemList[0]), float(itemList[1]), float(itemList[2]), float(itemList[3]), float(itemList[4]), int(itemList[5])) )
  return p,pid

def Evaluation2(trialname):
  
  #相互推定の学習結果データを読み込む
  #MI_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  #ARI_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  #PARs_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  #PARw_List = [[0.0 for i in xrange(R)] for j in xrange(step)]
  #PRR_List = [[0.0 for i in xrange(R)] for j in xrange(step)] 
  # location accuracy rate from a name of place 
  
  MAX_Samp = 0
  
  L = [[0.0 for i in xrange(R)] for j in xrange(step)]
  K = [[0.0 for i in xrange(R)] for j in xrange(step)]
  

  
  #相互推定のイテレーションと単語分割結果の候補のすべてのパターンの評価値を保存
  #fp_ARI = open('./data/' + filename + '/' + filename + '_A_sougo_ARI.csv', 'w')  
  #fp_PARs = open('./data/' + filename + '/' + filename + '_A_sougo_PARs.csv', 'w')  
  #fp_PARw = open('./data/' + filename + '/' + filename + '_A_sougo_PARw.csv', 'w')  
  #fp_MI = open('./data/' + filename + '/' + filename + '_A_sougo_MI.csv', 'w')  
  #fp_PRR = open(datafolder + trialname + '/' + trialname + '_meanEvaluationPRR2.csv', 'w')  
  #fp.write('MI,ARI,PARs,PARw\n')
  #fp.write('PRR\n')
  
  i = 0
  #重みファイルを読み込み
  for line in open(datafolder + trialname + '/'+ str(50) + '/weights.csv', 'r'):   ##読み込む
        #itemList = line[:-1].split(',')
        if (i == 0):
          MAX_Samp = int(line)
        i += 1
        
  #最大尤度のパーティクル番号を保存
  maxparticle = MAX_Samp
  
  particle,pid = ReadParticleData3(50, maxparticle, trialname)# [0 for i in range(50)] 
  
  #相互推定のイテレーションごとに
  for s in xrange(step):
      r = pid[s]
      #イテレーションごとに選ばれた学習結果の評価値をすべて保存するファイル
      fp = open(datafolder + trialname + '/' + str(s+1) + '/Stlit3' + str(r) + '.csv', 'w')
      
      
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
      
      #p_stlit = [[0.0 for w in range(len(W_index))] for i in range(K)]
      
      p_stlit = Location_from_speech(s+1, trialname, THETA, r,  L[s][r], K[s][r])
      
      print "OK!"
      for i in range(K[s][r]):
        for w in range(len(W_index)):
          fp.write(str(p_stlit[i][w]))
          #fp_PRR.write(str( PRR_List[s][r] ))
          fp.write(',')
    
        fp.write('\n')
      fp.close()
    
  print "close."
  
  
  
if __name__ == '__main__':
    #出力ファイル名を要求
    trialname = raw_input("trialname?(folder) >") #"tamd2_sig_mswp_01" 
    
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
    #for s in range(1,11): #１からstepの数字までのループ
    Evaluation2(trialname)# + str(s).zfill(3))
