#coding:utf-8

##############################################
##場所概念と言語モデルの相互推定モデル(TAMD2用)
##単語の選択：バイナリ相互情報量
##単語分割結果選択：CtとOtbの相互情報量

#####ROSで取った位置座標と教示データを読み込む
#####教示データ読み込み→学習→描画（座標系変換）
#####背景画像にMAPの画像を読み込む。
#####位置情報のデータと単語分割データの並び順は統一していること。
#####読み込んだ位置データを描画
##############################################


#---遂行タスクI(TAMD)---#

##↓別プログラムとして作成中
#相互推定のイテレーションごとに選ばれた学習結果の評価値（ARI、コーパスPAR、単語PAR）、（事後確率値）を出力
##単語PARはp(O_best|x_t)と正解を比較。x_tは正しいデータの平均値とする。

#↑のプログラムをインポートする

#---作業終了タスク（TAMD）---#
#sig_initパラメータを＿init＿.pyへ
#単語分割結果の読み込みの処理を修正
###pygameをインポート
#相互推定のイテレーションごとに位置分布の描画、保存
#データファイルを__init__.pyで指定するようにした
#パラメータ合わせる
#Turtlebotのデータに対応させる
#プログラムの整理（ちょっとだけ、無駄な計算や引数宣言の消去）
#sum()がnp.arrayの場合np.sumの方が高速(?)のため変更
#単語辞書が日本語音節表記でも音素ローマ字表記でも対応可能
#位置分布のμの値を保存するときに余計なものが混入していた（"_*"）のを修正済み

#---作業終了タスク---#
###正規化処理、0ワリ回避処理のコード短縮
###2次元ガウス分布の関数化、共分散をnumpy形式の行列計算に変更(時間があれば再確認)
#通常のMCLを平滑化MCLにする(結果確認済)
### Julius機能を消去。教示モードは動作をしながら教示時刻のみを保存する。(要確認)
#range()をxrange()に全て変更。xrange()の方が計算効率が良いらしい。
##ラグ値(LAG,lagu)、普通のMCL(ラグ値=0)の平滑化結果の出力を一つだけ(LAG)にする
##動作モデルと計測モデルの関数化->計測モデルの簡素化
###教示フェーズとMCLフェーズをわける
###センサ値、制御値を保存する
###保存したデータから平滑化MCLを動作させる
##全ての処理終了後に平滑化自己位置推定の結果をファイル出力(csv)
##認識発話単語集合をファイルへ出力
#パーティクル初期化の方法を変更(リスト内包表記いちいちしない)->いちいちリスト内表記！
##発話認識文(単語)データを読み込む
#<s>,<sp>,</s>を除く処理
#角度からラジアンへ変換する関数radian()はmathにあるためそちらに変更
##stick_breaking関数を見つけたので入れてみた
#多項分布の確率質量関数pmfを計算する関数を発見したので導入
##位置分布の描画用処理、viewerに送る位置分布サンプリング点群itiを設定
#robot初期状態を表す(x_init,y_init,radians(d_init))を作った
#motion_modelの関数名をsample_motion_modelへ変更。
#パーティクル数Mと位置分布平均Myuの引数名区別。Myuの初期化法修正
#0ワリ対処関数yudoupにyudo_sum==0の場合の例外処理を加えた。
###sampleではないmotion_modelを実装
#角度処理関数kakudoはPIの小数点精度問題があったため、より精度のよい修正版のkakudo2を作成。
#####XDt_true(真の角度値)とXDtが一致しない件について調査(角度が一周してることが判明)
######SBP,weak limit approximationについての確認、SBP他、初期化方法の修正
#myu0->m0,VV0->V0に修正
##動かさずにサンプル散らす関数sample_not_motion_modelを作った(挙動の確認する必要がある)
##最終学習結果の出力：初期値(*_init.csv)およびイテレーションごとにファイル出力
###データののっている要素のみをプリントする（データなしは表示しないようにしたが、ファイル出力するときは全部出す）
#パーティクルの平均を求める部分のコード短縮（ある程度できた）
#各サンプリングにおいて、データのない番号のものはどうする？：消す、表示しない、番号を前倒しにするか等
###各サンプリングにおいて、正しく正規化処理が行われているかチェック->たぶんOK
#####motion_modelの角度の例外処理->一応とりあえずやった
###ギブスサンプリングの収束判定条件は？(イテレートを何回するか)->とりあえず100回
###初期パラメータの値はどうするか？->とりあえずそこそこな感じにチューニングした
###どの要素をどの順番でサンプリングするか？->現状でとりあえずOK

#---保留---#
#計算速度の効率化は後で。
#いらないものは消す->少しは消した
##motionmodelで、パーティクルごとにおくるのではなく、群一気に行列としておくってnumpyで行列演算したほうが早いのでは？
##↑センサーモデルも同様？
#NormalInverseWishartDistribution関数のプログラムを見つけた。正確かどうか、どうなってるのか、要調査。
#余裕があれば場所概念ごとに色分けして位置分布を描画する(場所概念ごとの混合ガウス)
#Xtとμが遠いとg2の値がアンダーフローする可能性がある(logで計算すればよい？)問題があれば修正。
#ガウス分布を計算する関数をlogにする
#センサ値をintにせずにそのまま利用すればセンサ関係の尤度の値の計算精度があがるかも？
###動作モデルがガウスなので計算で求められるかもしれない件の数式導出


##ギブスサンプリング##
#W～ディリクレ＝マルチ*ディリクレ  L個：実装できたかな？
#μ、Σ～ガウス*ガウスウィシャート  K個：旧モデルの流用でok
#π～ディリクレ＝マルチ*GEM  1個：一応できた？GEM分布の事後分布の計算方法要確認
#Φ～ディリクレ＝マルチ*GEM  L個：同上
#Ct～多項値P(O|Wc)*多項値P(i|φc)*多項P(c|π)  N個：できた？
#it～ガウス値N(x|μk,Σk)*多項P(k|φc)  N個：できた？
#xt(no t)～計測モデル値*動作モデル値*動作モデルパーティクル (EndStep-N)個：概ねできた？
#xt(on t)～計測モデル値*動作モデル値*itの式(混合ガウス値)*動作モデルパーティクル N個：同上

import glob
import codecs
import re
import os
import sys
import pygame
import random
import string
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
from pygame.locals import *
pygame.init()

from __init__ import *

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
    
def yudoup(yudo,yudo_sum): #float( 10 ** (-200) )
    if yudo_sum == 0 :  #エラー処理
        yudo = [0.1 for j in xrange(len(yudo))]
        yudo_sum = sum(yudo)
        print "yudo_sum is 0"
    if yudo_sum < 10**(-15) : #0.000000000000001: #+0000000000
        for j in xrange(len(yudo)):
          yudo[j] = yudo[j] * 10.0**12 #100000000000 #+00000
        yudo_sum = yudo_sum * 10.0**12 #100000000000 #+00000
        yudo,yudo_sum = yudoup(yudo,yudo_sum)
        print "yudoup!"
    return yudo,yudo_sum

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
"""
class NormalInverseWishartDistribution(object):
#http://stats.stackexchange.com/questions/78177/posterior-covariance-of-normal-inverse-wishart-not-converging-properly
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = float(lmbda)
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)

    def sample(self):
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
"""
"""
def motion_model(drot1,dtrans,para,pxt,pyt,pdt,pxt_1,pyt_1,pdt_1):
    #動作モデル(表5.5)##
    drot2 = 0  #0と仮定
    
    drot1_hat = atan2(pyt - pyt_1, pxt - pxt_1) - pdt_1
    dtrans_hat = sqrt( (pxt_1 - pxt)*(pxt_1 - pxt) + (pyt_1 - pyt)*(pyt_1 - pyt) )
    drot2_hat = pdt - pdt_1 - drot1_hat
    #角度処理：角度の差が[-PI,PI]以内であることが必要らしい
    
    #gaussian(x,myu,sig)
    p1 = gaussian(drot1 - drot1_hat, 0.0, para[1] * drot1_hat * drot1_hat + para[2] * dtrans_hat * dtrans_hat)
    p2 = gaussian(dtrans - dtrans_hat, 0.0, para[3] * dtrans_hat * dtrans_hat + para[4] * drot1_hat * drot1_hat + para[4] * drot2_hat * drot2_hat)
    p3 = gaussian(drot2 - drot2_hat, 0.0, para[1] * drot2_hat * drot2_hat + para[2] * dtrans_hat * dtrans_hat)
    
    return p1*p2*p3

def sample_motion_model(drot1,dtrans,para,ppx,ppy,pardirection):
    #動作モデル(表5.6)##
    drot2 = 0  #0と仮定
    #print u"%s %s" % (str(d_rot),str(d_trans))
    
    drot1_hat = drot1 - random.gauss(0,para[1] * drot1 * drot1 + para[2] * dtrans * dtrans)
    dtrans_hat = dtrans - random.gauss(0,para[3] * dtrans * dtrans + para[4] * drot1 * drot1 + para[4] * drot2 * drot2)
    drot2_hat = drot2 - random.gauss(0,para[1] * drot2 * drot2 + para[2] * dtrans * dtrans)
    
    xd = ppx + dtrans_hat * cos (pardirection + drot1_hat)
    yd = ppy + dtrans_hat * sin (pardirection + drot1_hat)
    sitad = pardirection + drot1_hat + drot2_hat
    
    #print u"%s %s %s" % (str(xd),str(pdrot_hat),str(drot1_hat))
    return xd,yd,kakudo2(sitad)
    
def sample_not_motion_model(drot1,dtrans,para,ppx,ppy,pardirection):
    #動かない動作モデル##
    drot2 = 0  #0と仮定
    #print u"%s %s" % (str(d_rot),str(d_trans))
    
    drot1_hat =  random.gauss(0,para[1] * drot1 * drot1 + para[2] * dtrans * dtrans)
    dtrans_hat =  random.gauss(0,para[3] * dtrans * dtrans + para[4] * drot1 * drot1 + para[4] * drot2 * drot2)
    drot2_hat =  random.gauss(0,para[1] * drot2 * drot2 + para[2] * dtrans * dtrans)
    
    xd = ppx + dtrans_hat * cos (pardirection + drot1_hat)
    yd = ppy + dtrans_hat * sin (pardirection + drot1_hat)
    sitad = pardirection + drot1_hat + drot2_hat
    
    return xd,yd,kakudo2(sitad)

def sensor_model(robot,xd,yd,sitad,sig_hit,values):
    #計測モデル：尤度(重み)計算##
    #各パーティクルにロボット飛ばす->センサー値を取得
    #新たに設定されたパーティクルへ一時的にロボットを送る
    robot.set_position(xd,yd,sitad)
    robot.input ((0,0))
    robot.move  (0, 0)
    
    q = 1.0
    #pvalues = []
    for j in xrange(len(robot.sensors)):
        ss = robot.sensors[j]
        #pvalues = pvalues +  [int((1.0-ss.value)*ss.limit/2.0/5)]   ###各センサー値を格納
        gauss_s = gaussian(int((1.0-ss.value)*ss.limit/2.0/5), values[j], sig_hit)  ###ガウス分布
        
        #q = q * fabs( gaussian ) 
        q = q + ( fabs( gauss_s ) * 800.0 + fabs( random.gauss(0,3) ) )   #元の設定
        #q = q * ( fabs( gaussian ) * 800000000 + fabs( random.gauss(0,10**(-15)) ) )
        ###print u"dsumt:%f distsum: %f q: %f" % (dsumt,distsum,q)
    return q
"""
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

def MI_binary(b,W,pi,c):  #Mutual information(二値版):word_index、W、π、Ct
    #相互情報量の計算
    POC = W[c][b] * pi[c] #Multinomial(W[c]).pmf(B) * pi[c]   #場所の名前の多項分布と場所概念の多項分布の積
    PO = sum([W[ct][b] * pi[ct] for ct in xrange(L)]) #Multinomial(W[ct]).pmf(B)
    PC = pi[c]
    POb = 1.0 - PO
    PCb = 1.0 - PC
    PObCb = PCb - PO + POC
    POCb = PO - POC
    PObC = PC - POC
    
    # 相互情報量の定義の各項を計算
    temp1 = POC * log(POC/(PO*PC), 2)
    temp2 = POCb * log(POCb/(PO*PCb), 2)
    temp3 = PObC * log(PObC/(POb*PC), 2)
    temp4 = PObCb * log(PObCb/(POb*PCb), 2)
    score = temp1 + temp2 + temp3 + temp4
    return score

def Mutual_Info(W,pi):  #Mutual information:W、π 
    MI = 0
    for c in xrange(len(pi)):
      PC = pi[c]
      for j in xrange(len(W[c])):
        #B = [int(i==j) for i in xrange(len(W[c]))]
        PO = fsum([W[ct][j] * pi[ct] for ct in xrange(len(pi))])  #Multinomial(W[ct]).pmf(B)
        POC = W[c][j] * pi[c]   #場所の名前の多項分布と場所概念の多項分布の積
        
        
        # 相互情報量の定義の各項を計算
        MI = MI + POC * ( log((POC/(PO*PC)), 2) )
    
    return MI


# 定数
WINDOW_SIZE = (WallX,WallY)
FPS = 60

# 色
BLACK = (0,0,0)
WHITE = (255,255,255)
#RED = (255,0,255)
CYAN = (0,255,255)

# 描画、物体定義・当たり判定・センサー値計算 は 他のモジュールに任せる
# usage:
#   1. world, viewer(pygameを渡す)の初期化
#   2. loop
#      2.1 world  の関数を使って 入力, 更新
#      2.2 viewer に描画情報(world,fps)渡して更新
import viewer
import world
class Input:
    def __init__(self,pygame):
        self.pygame = pygame
        try:
            self.js = self.pygame.joystick.Joystick(0)
            self.js.init()
            #print 'Recognize Joystick.'
            self.flag = True
        except self.pygame.error:
            #print 'Cannot find Joystick.'
            print 'use keyboard.'
            self.flag = False
    def get_AXIS(self):
        self.LAXIS_X = self.js.get_axis(0)
        self.LAXIS_Y = self.js.get_axis(1)
        self.RAXIS_X = self.js.get_axis(2)
        self.RAXIS_Y = self.js.get_axis(3)
    def get_value(self,events):
        if self.flag:
            self.get_AXIS()
            return [self.LAXIS_Y,self.RAXIS_Y]


#位置推定の教示データファイル名を要求
#data_name = raw_input("Read_XTo_filename?(.csv) >")


# 初期化
viewer=viewer.Viewer(WINDOW_SIZE,BLACK,'MAP Viewer')
world .init()
input = Input(pygame)
MAINCLOCK = pygame.time.Clock() # fps調整用

## 物体の設定
# 場所１
place = world.Place()

"""
# 外壁
place.add_wall (    0,    0,WallX,    0,WHITE)
place.add_wall (    0,    0,    0,WallY,WHITE)
place.add_wall (WallX,    0,WallX,WallY,WHITE)
place.add_wall (    0,WallY,WallX,WallY,WHITE)
"""


# worldに追加, crnt_placeを設定
world.add_place(place)
world.change_place(0)

# ロボット１
robot = world.Robot()
x_init,y_init,d_init = 140,240,350
#Xinit = (x_init,y_init,radians(d_init))  #ロボットの初期状態
robot.set_position(x_init,y_init,radians(d_init))
robot.set_appearance(WHITE,10)
robot.set_bias(2,2)
#for i in xrange(10):
#    robot.add_sensor(-50+i*2.5*2,150,WHITE,3)
#for i in xrange(10):
#    robot.add_sensor( i*2.5*2,150,WHITE,3)

# world に追加
world.add_robot(robot)


# Simulation
def simulate(iteration):
    #viewer.show(world,[[0,0]],0,[],[])
    
    #learning = 1    #学習タスク=1，修正タスク=0(未使用)
    #kyouji_count = 0 #教示数をカウントする
    #EndStep = 0    #教示タスク終了時刻(step)を記録
    #TN = []          #教示された時間を保存する
    
    
    
    #kyouji_count = 40 #########################################################
    #EndStep = 40-1    #########################################################
    
    ##発話認識文(単語)データを読み込む
    ##空白またはカンマで区切られた単語を行ごとに読み込むことを想定する
    
    #sample_num = 1  #取得するサンプル数
    #N = 0      #データ個数用
    #Otb = [[] for sample in xrange(sample_num)]   #音声言語情報：教示データ
    
    for sample in xrange(sample_num):
      #NN = 0
      N = 0
      Otb = []
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
        
        #Otb[sample] = Otb[sample] + [itemList]
        Otb = Otb + [itemList]
        #if sample == 0:  #最初だけデータ数Nを数える
        N = N + 1  #count
        #else:
        #  Otb[] = Otb[NN] + itemList
        #  NN = NN + 1
        
        for j in xrange(len(itemList)):
            print "%s " % (str(itemList[j])),
        print ""  #改行用
      
      
      ##場所の名前の多項分布のインデックス用
      W_index = []
      for n in xrange(N):
        for j in xrange(len(Otb[n])):
          if ( (Otb[n][j] in W_index) == False ):
            W_index.append(Otb[n][j])
            #print str(W_index),len(W_index)
      
      print "[",
      for i in xrange(len(W_index)):
        print "\""+ str(i) + ":" + str(W_index[i]) + "\",",
      print "]"
      
      ##時刻tデータごとにBOW化(?)する、ベクトルとする
      Otb_B = [ [0 for i in xrange(len(W_index))] for n in xrange(N) ]
      
      
      for n in xrange(N):
        for j in xrange(len(Otb[n])):
          for i in xrange(len(W_index)):
            if (W_index[i] == Otb[n][j] ):
              Otb_B[n][i] = Otb_B[n][i] + 1
      #print Otb_B
      
      
      if kyouji_count != N:
         print "N:KYOUJI error!!" + str(N)   ##教示フェーズの教示数と読み込んだ発話文データ数が違う場合
         #exit()
      
      #TN = [i for i in xrange(N)]#[0,1,2,3,4,5]  #テスト用
      
      ##教示位置をプロットするための処理
      #x_temp = []
      #y_temp = []
      #for t in xrange(len(TN)):
      #  x_temp = x_temp + [Xt[int(TN[t])][0]]  #設定は実際の教示時刻に対応できるようになっている。
      #  y_temp = y_temp + [Xt[int(TN[t])][1]]  #以前の設定のままで、動かせるようにしている。
      
      if (data_name != 'test000'):
        i = 0
        Xt = []
        #Xt = [(0.0,0.0) for n in xrange(len(HTW)) ]
        TN = []
        for line3 in open('./../sample/' + data_name, 'r'):
          itemList3 = line3[:-1].split(' ')
          Xt = Xt + [(float(itemList3[0]), float(itemList3[1]))]
          TN = TN + [i]
          print TN
          i = i + 1
        
        #Xt = Xt_temp
        EndStep = len(Xt)-1
      
      else:
        ###SIGVerse###
        HTW = []
        for line2 in open('./../sample/' + data_name +  '_HTW.csv', 'r'):
          itemList2 = line2[:-1].split(',')
          HTW = HTW + [itemList2[0]]
        
        i = 0
        Xt_temp = []
        Xt = [(0.0,0.0) for n in xrange(len(HTW)) ]
        TN = []
        for line3 in open('./../sample/' + data_name +  '_X_To.csv', 'r'):
          itemList3 = line3[:-1].split(',')
          Xt_temp = Xt_temp + [(float(itemList3[2]) + 500, float(itemList3[1]) + 250)]
          TN = TN + [i]
          print TN
          i = i + 1
        
        #げんかん(ge)0-9、てえぶるのあたり黒(teb)10-19、白(tew)20-29、ほんだな(hd)30-39、
        #そふぁあまえ(sf)40-49、きっちん(kt)50-59、だいどころ(dd)60-69、ごみばこ(go)70-79、てれびまえ(tv)80-89
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
            
        
        EndStep = len(Xt)-1
        
        #x_temp = []
        #y_temp = []
        #for t in xrange(len(Xt)):
        #  #x_temp = x_temp + [Xt[int(TN[t])][0]]
        #  #y_temp = y_temp + [Xt[int(TN[t])][1]]
        #  x_temp = x_temp + [Xt[t][0]]
        #  y_temp = y_temp + [Xt[t][1]]
        ###SIGVerse###
      
      #x_temp = []
      #y_temp = []
      
      #x_temp = [Xt[t][0] for t in xrange(len(Xt))]
      #y_temp = [Xt[t][1] for t in xrange(len(Xt))]
      
      #for t in xrange(len(Xt)):
      #  #x_temp = x_temp + [Xt[int(TN[t])][0]]
      #  #y_temp = y_temp + [Xt[int(TN[t])][1]]
      #  x_temp = x_temp + [Xt[t][0]]
      #  y_temp = y_temp + [Xt[t][1]]
      #  
      #events = pygame.event.get()
      #robot.input ((0,0))
      #robot.move  (0, 0)
      #viewer.show(world,[[0,0]],len(TN),x_temp,y_temp)
      
  ######################################################################
  ####                   ↓場所概念学習フェーズ↓                   ####
  ######################################################################
      #TN[N]：教示時刻(step)集合
      
      #Otb_B[N][W_index]：時刻tごとの発話文をBOWにしたものの集合
      
      ##各パラメータ初期化処理
      print u"Initialize Parameters..."
      #xtは既にある、ct,it,Myu,S,Wは事前分布からサンプリングにする？(要相談)
      Ct = [ int(random.uniform(0,L)) for n in xrange(N)] #[0,0,1,1,2,3]     #物体概念のindex[N]
      It = [ int(random.uniform(0,K)) for n in xrange(N)]#[1,1,2,2,3,2]     #位置分布のindex[N]
      ##領域範囲内に一様乱数
      if (data_name == "test000"):
        Myu = [ np.array([[ int( random.uniform(1,WallX-1) ) ],[ int( random.uniform(1,WallY-1) ) ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
      else:
        Myu = [ np.array([[ random.uniform(-37.8+5,-37.8+80-10) ],[ random.uniform(-34.6+5,-34.6+57.6-10) ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
      S = [ np.array([ [sig_init, 0.0],[0.0, sig_init] ]) for i in xrange(K) ]      #位置分布の共分散(2×2次元)[K]
      W = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
      pi = stick_breaking(gamma, L)#[ 0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
      phi_l = [ stick_breaking(alpha, K) for c in xrange(L) ]#[ [0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K次元)[L]
      
      
      print Myu
      print S
      print W
      print pi
      print phi_l
      
      ###初期値を保存(このやり方でないと値が変わってしまう)
      Ct_init = [Ct[n] for n in xrange(N)]
      It_init = [It[n] for n in xrange(N)]
      Myu_init = [Myu[i] for i in xrange(K)]
      S_init = [ np.array([ [S[i][0][0], S[i][0][1]],[S[i][1][0], S[i][1][1]] ]) for i in xrange(K) ]
      W_init = [W[c] for c in xrange(L)]
      pi_init = [pi[c] for c in xrange(L)]
      phi_l_init = [phi_l[c] for c in xrange(L)]
      
      
      
      
      ##場所概念の学習
      #関数にとばす->のは後にする
      print u"- <START> Learning of Location Concepts ver. NEW MODEL. -"
      
      for iter in xrange(num_iter):   #イテレーションを行う
        print 'Iter.'+repr(iter+1)+'\n'
        
        
        ########## ↓ ##### it(位置分布のindex)のサンプリング ##### ↓ ##########
        print u"Sampling it..."
        
        #It_B = [0 for k in xrange(K)] #[ [0 for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #itと同じtのCtの値c番目のφc  の要素kごとに事後多項分布の値を計算
        temp = np.zeros(K)
        for t in xrange(N):    #時刻tごとのデータ
          phi_c = phi_l[int(Ct[t])]
          #np.array([ 0.0 for k in xrange(K) ])   #多項分布のパラメータ
          
          for k in xrange(K):
            #phi_temp = Multinomial(phi_c)
            #phi_temp.pmf([kのとき1のベクトル]) #パラメータと値は一致するのでphi_c[k]のままで良い
            
            #it=k番目のμΣについてのガウス分布をitと同じtのxtから計算
            xt_To = TN[t]
            g2 = gaussian2d(Xt[xt_To][0],Xt[xt_To][1],Myu[k][0],Myu[k][1],S[k])  #2次元ガウス分布を計算
            
            temp[k] = g2 * phi_c[k]
            #print g2,phi_c[k]  ###Xtとμが遠いとg2の値がアンダーフローする可能性がある
            
          temp = temp / np.sum(temp)  #正規化
          #print temp
          #Mult_samp = np.random.multinomial(1,temp)
          
          #print Mult_samp
          It_B = np.random.multinomial(1,temp) #Mult_samp [t]
          #print It_B[t]
          It[t] = np.where(It_B == 1)[0][0] #It_B.index(1)
          #for k in xrange(K):
          #  if (It_B[k] == 1):
          #    It[t] = k
          #    #print k
          
        #gaussian2d(Xx,Xy,myux,myuy,sigma)
        
        print It
        
        #多項分布からのサンプリング(1点)
        #http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html#numpy.random.multinomial
        #Mult_samp = np.random.multinomial(1,[確率の配列])
        ########## ↑ ##### it(位置分布のindex)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### Ct(場所概念のindex)のサンプリング ##### ↓ ##########
        print u"Sampling Ct..."
        #Ct～多項値P(Ot|Wc)*多項値P(it|φc)*多項P(c|π)  N個
        
        #It_B = [ [int(k == It[n]) for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #Ct_B = [0 for c in xrange(L)] #[ [0 for c in xrange(L)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][l]
        
        temp = np.zeros(L)
        for t in xrange(N):    #時刻tごとのデータ
          #for k in xrange(K):
          #  if (k == It[t]):
          #    It_B[t][k] = 1
          
          #print It_B[t] #ok
          
          #np.array([ 0.0 for c in xrange(L) ])   #多項分布のパラメータ
          for c in xrange(L):  #場所概念のindexの多項分布それぞれについて
            #phi_temp = Multinomial(phi_l[c])
            W_temp = Multinomial(W[c])
            #print pi[c], phi_temp.pmf(It_B[t]), W_temp.pmf(Otb_B[t])
            temp[c] = pi[c] * phi_l[c][It[t]] * W_temp.pmf(Otb_B[t])    # phi_temp.pmf(It_B[t])各要素について計算
          
          temp = temp / np.sum(temp)  #正規化
          #print temp
          #Mult_samp = np.random.multinomial(1,temp)
          
          #print Mult_samp
          Ct_B = np.random.multinomial(1,temp) #Mult_samp
          #print Ct_B[t]
          
          Ct[t] = np.where(Ct_B == 1)[0][0] #Ct_B.index(1)
          #for c in xrange(L):
          #  if (Ct_B[c] == 1):
          #    Ct[t] = c
          #    #print c
          
        print Ct
        ########## ↑ ##### Ct(場所概念のindex)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### W(場所の名前：多項分布)のサンプリング ##### ↓ ##########
        ##ディリクレ多項からディリクレ事後分布を計算しサンプリングする
        ##ディリクレサンプリング関数へ入れ込む配列を作ればよい
        ##ディリクレ事前分布をサンプリングする必要はない->共役
        print u"Sampling Wc..."
        
        #data = [Otb_B[1],Otb_B[3],Otb_B[7],Otb_B[8]]  #仮データ
        
        #temp = np.ones((len(W_index),L))*beta0 #
        temp = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #集めて加算するための配列:パラメータで初期化しておけばよい
        #temp = [ np.ones(len(W_index))*beta0 for c in xrange(L)]
        #Ctがcであるときのデータを集める
        for c in xrange(L) :   #ctごとにL個分計算
          #temp = np.ones(len(W_index))*beta0
          nc = 0
          ##事後分布のためのパラメータ計算
          if c in Ct : 
            for t in xrange(N) : 
              if Ct[t] == c : 
                #データを集めるたびに値を加算
                for j in xrange(len(W_index)):    #ベクトル加算？頻度
                  temp[c][j] = temp[c][j] + Otb_B[t][j]
                nc = nc + 1  #データが何回加算されたか
              
          if (nc != 0):  #データなしのcは表示しない
            print "%d n:%d %s" % (c,nc,temp[c])
          
          #加算したデータとパラメータから事後分布を計算しサンプリング
          sumn = sum(np.random.dirichlet(temp[c],1000)) #fsumではダメ
          W[c] = sumn / sum(sumn)
          #print W[c]
        
        #Dir_0 = np.random.dirichlet(np.ones(L)*jp)
        #print Dir_0
        
        #ロバストなサンプリング結果を得るために
        #sumn = sum(np.random.dirichlet([0.1,0.2,0.5,0.1,0.1],10000))
        #multi = sumn / fsum(sumn)
        
        ########## ↑ ##### W(場所の名前：多項分布)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### μΣ(位置分布：ガウス分布の平均、共分散行列)のサンプリング ##### ↓ ##########
        print u"Sampling myu_i,Sigma_i..."
        #myuC = [ np.zeros((2,1)) for k in xrange(K) ] #np.array([[ 0.0 ],[ 0.0 ]])
        #sigmaC = [ np.zeros((2,2)) for k in xrange(K) ] #np.array([ [0,0],[0,0] ])
        np.random.seed()
        nk = [0 for j in xrange(K)]
        for j in xrange(K) : 
          ###jについて、Ctが同じものを集める
          #n = 0
          
          xt = []
          if j in It : 
            for t in xrange(N) : 
              if It[t] == j : 
                xt_To = TN[t]
                xt = xt + [ np.array([ [Xt[xt_To][0]], [Xt[xt_To][1]] ]) ]
                nk[j] = nk[j] + 1
          
          m_ML = np.array([[0.0],[0.0]])
          if nk[j] != 0 :        ##0ワリ回避
            m_ML = sum(xt) / float(nk[j]) #fsumではダメ
            print "n:%d m_ML.T:%s" % (nk[j],str(m_ML.T))
          
          #m0 = np.array([[0],[0]])   ##m0を元に戻す
          
          ##ハイパーパラメータ更新
          kappaN = kappa0 + nk[j]
          mN = ( (kappa0*m0) + (nk[j]*m_ML) ) / kappaN
          nuN = nu0 + nk[j]
          
          dist_sum = 0.0
          for k in xrange(nk[j]) : 
            dist_sum = dist_sum + np.dot((xt[k] - m_ML),(xt[k] - m_ML).T)
          VN = V0 + dist_sum + ( float(kappa0*nk[j])/(kappa0+nk[j]) ) * np.dot((m_ML - m0),(m_ML - m0).T)
          
          #if nk[j] == 0 :        ##0ワリ回避
          #  #nuN = nu0# + 1  ##nu0=nuN=1だと何故かエラーのため
          #  #kappaN = kappaN# + 1
          #  mN = np.array([[ int( random.uniform(1,WallX-1) ) ],[ int( random.uniform(1,WallY-1) ) ]])   ###領域内に一様
          
          ##3.1##Σを逆ウィシャートからサンプリング
          
          samp_sig_rand = np.array([ invwishartrand(nuN,VN) for i in xrange(100)])    ######
          samp_sig = np.mean(samp_sig_rand,0)
          #print samp_sig
          
          if np.linalg.det(samp_sig) < -0.0:
            samp_sig = np.mean(np.array([ invwishartrand(nuN,VN)]),0)
          
          ##3.2##μを多変量ガウスからサンプリング
          #print mN.T,mN[0][0],mN[1][0]
          x1,y1 = np.random.multivariate_normal([mN[0][0],mN[1][0]],samp_sig / kappaN,1).T
          #print x1,y1
          
          Myu[j] = np.array([[x1],[y1]])
          S[j] = samp_sig
          
        
        for j in xrange(K) : 
          if (nk[j] != 0):  #データなしは表示しない
            print 'myu'+str(j)+':'+str(Myu[j].T),
        print ''
        
        for j in xrange(K):
          if (nk[j] != 0):  #データなしは表示しない
            print 'sig'+str(j)+':'+str(S[j])
          
          
        """
        #データのあるKのみをプリントする？(未実装)
        print "myu1:%s myu2:%s myu3:%s myu4:%s myu5:%s" % (str(myuC[0].T), str(myuC[1].T), str(myuC[2].T),str(myuC[3].T), str(myuC[4].T))
        print "sig1:\n%s \nsig2:\n%s \nsig3:\n%s" % (str(sigmaC[0]), str(sigmaC[1]), str(sigmaC[2]))
        """
        #Myu = myuC
        #S = sigmaC
        
        ########## ↑ ##### μΣ(位置分布：ガウス分布の平均、共分散行列)のサンプリング ##### ↑ ##########
        
        
       ########## ↓ ##### π(場所概念のindexの多項分布)のサンプリング ##### ↓ ##########
        print u"Sampling PI..."
        
        #GEM = stick_breaking(gamma, L)
        #print GEM
        
        temp = np.ones(L) * (gamma / float(L)) #np.array([ gamma / float(L) for c in xrange(L) ])   #よくわからないので一応定義
        for c in xrange(L):
          temp[c] = temp[c] + Ct.count(c)
        #for t in xrange(N):    #Ct全データに対して
        #  for c in xrange(L):  #index cごとに
        #    if Ct[t] == c :      #データとindex番号が一致したとき
        #      temp[c] = temp[c] + 1
        #print temp  #確認済み
        
        #とりあえずGEMをパラメータとして加算してみる->桁落ちが発生していて意味があるのかわからない->パラメータ値を上げてみる&tempを正規化して足し合わせてみる(やめた)
        #print fsum(GEM),fsum(temp)
        #temp = temp / fsum(temp)
        #temp =  temp + GEM
        
        #持橋さんのスライドのやり方の方が正しい？ibis2008-npbayes-tutorial.pdf
        
        #print temp
        #加算したデータとパラメータから事後分布を計算しサンプリング
        sumn = sum(np.random.dirichlet(temp,1000)) #fsumではダメ
        pi = sumn / np.sum(sumn)
        print pi
        
        ########## ↑ ##### π(場所概念のindexの多項分布)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### φ(位置分布のindexの多項分布)のサンプリング ##### ↓ ##########
        print u"Sampling PHI_c..."
        
        #GEM = [ stick_breaking(alpha, K) for c in xrange(L) ]
        #print GEM
        
        for c in xrange(L):  #L個分
          temp = np.ones(K) * (alpha / float(K)) #np.array([ alpha / float(K) for k in xrange(K) ])   #よくわからないので一応定義
          #Ctとcが一致するデータを集める
          if c in Ct :
            for t in xrange(N):
              if Ct[t] == c:  #Ctとcが一致したデータで
                for k in xrange(K):  #index kごとに
                  if It[t] == k :      #データとindex番号が一致したとき
                    temp[k] = temp[k] + 1  #集めたデータを元に位置分布のindexごとに加算
            
          
          #ここからは一個分の事後GEM分布計算(πのとき)と同様
          #print fsum(GEM[c]),fsum(temp)
          #temp = temp / fsum(temp)
          #temp =  temp + GEM[c]
          
          #加算したデータとパラメータから事後分布を計算しサンプリング
          sumn = sum(np.random.dirichlet(temp,1000)) #fsumではダメ
          phi_l[c] = sumn / np.sum(sumn)
          
          if c in Ct:
            print c,phi_l[c]
          
          
        ########## ↑ ##### φ(位置分布のindexの多項分布)のサンプリング ##### ↑ ##########
        
        
        """
        ########## ↓ ##### xt(教示時刻で場合分け)のサンプリング ##### ↓ ##########
        print u"Sampling xt..."
        robot.input ((0,0))
        robot.move  (0, 0)
        
        
        #It_B = [ [0 for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #
        #for t in xrange(N):    #時刻tごとのデータ
        #  for k in xrange(K):
        #    if (k == It[t]):
        #      It_B[t][k] = 1
        
        #It_1 = [ [(i==j)*1 for i in xrange(L)] for j in xrange(L)]   #i==jの要素が1．それ以外は0のベクトル
        
        #for t in xrange(EndStep):
        t = -1#EndStep-1
        while (t >= 0):
          ##t in Toかどうか関係ない部分の処理
          Xx_temp,Xy_temp,Xd_temp = [],[],[]
          yudo = []
          
          input1,input2 = Ut[t][0],Ut[t][1]
          robot.input ((input1,input2))
          robot.move  (d_trans, d_rot)
          d_trans = input2 * robot.bias_turn      #
          d_rot = radians(robot.bias_go) * input1  #
          #print t
          if (t+1 < EndStep):
            d_trans2 = Ut[t+1][1] * robot.bias_turn      #
            d_rot2 = radians(robot.bias_go) * Ut[t+1][0]  #
          
          for i in xrange(M):   ##全てのパーティクルに対し
            #動作モデルによりt-1からtの予測分布をサンプリング
            #動作モデル(表5.6)##↓###################################################ok
            if (t == 0):
              #xd,yd,sitad = sample_motion_model(d_rot,d_trans,para,Xinit[0],Xinit[1],Xinit[2]) #初期値を与えてよいのか？
              #xd,yd,sitad = Xt[t][0],Xt[t][1],XDt[t]  #最初の推定結果をそのまま用いる場合
              xd,yd,sitad = sample_not_motion_model(d_rot,d_trans,para_s,Xt[t][0],Xt[t][1],XDt[t]) #動かさずに粒子を散らすだけ
            else:
              xd,yd,sitad = sample_motion_model(d_rot,d_trans,para_s,Xt[t-1][0],Xt[t-1][1],XDt[t-1])
            Xx_temp = Xx_temp + [xd]
            Xy_temp = Xy_temp + [yd]
            Xd_temp = Xd_temp + [sitad]
            #動作モデル##↑###################################################
            
            #計測モデルを計算
            #尤度(重み)計算##↓###########################################
            #ロボットの姿勢、センサー値(地図)、パーティクルのセンサー値(計測)
            #各パーティクルにロボット飛ばす->センサー値を取得 をパーティクルごとに繰り返す
            yudo = yudo + [sensor_model(robot,xd,yd,sitad,sig_hit2,Zt[t])]
            #尤度(重み)計算##↑###########################################
            
            
          ###一回正規化してから尤度かけるようにしてみる
          #正規化処理
          #yudo_sum = fsum(yudo)
          #yudo,yudo_sum = yudoup(yudo,yudo_sum)     ####とても小さな浮動小数値をある程度まで大きくなるまで桁をあげる
          ###0ワリ対処処理
          #yudo_max = max(yudo)  #最大尤度のパーティクルを探す
          #yudo_summax = float(yudo_sum) / yudo_max
          #for j in xrange(M):
          #  yudo[j] = float(float(yudo[j])/yudo_max) / yudo_summax
          #  
          #  
          #for i in xrange(M):   ##全てのパーティクルに対し
            
            
            #動作モデル(t+1)尤度計算
            if (t+1 < EndStep):
              #print yudo[i],motion_model(d_rot2,d_trans2,para,Xt[t+1][0],Xt[t+1][1],XDt[t+1],xd,yd,sitad)
              yudo[i] = yudo[i] * motion_model(d_rot2,d_trans2,para_s,Xt[t+1][0],Xt[t+1][1],XDt[t+1],xd,yd,sitad)
            
            #tによって場合分け処理
            for n in xrange(N):
              if TN[n] == t:  #t in To
                #ガウス×多項 / Σ(ガウス×多項)-> ガウス / Σ(ガウス×多項)
                GM_sum = 0.0
                #print t
                #分母：混合ガウス部分の計算
                #phi_temp = Multinomial(phi_l[Ct[n]])
                for j in xrange(K):  #it=jごとのすべての位置分布において
                  ##パーティクルごとに計算する必要がある、パーティクルごとに値をもっていないといけない？
                  
                  g2 = gaussian2d(xd,yd,Myu[j][0],Myu[j][1],S[j])  #2次元ガウス分布を計算
                  GM_sum = GM_sum + g2 * phi_l[Ct[n]][j]    #各要素について計算
                  #phi_temp.pmf( It_1[j] )
                  
                  ##
                if (GM_sum != 0):
                  yudo[i] = yudo[i] * gaussian2d(xd,yd,Myu[It[n]][0],Myu[It[n]][1],S[It[n]]) / GM_sum
                #print yudo[i]
            
            
          ##推定状態確認用
          #MAINCLOCK.tick(FPS)
          events = pygame.event.get()
          for event in events:
                if event.type == KEYDOWN:
                    if event.key  == K_ESCAPE: exit()
          robot.set_position(Xt_true[t][0],Xt_true[t][1],XDt_true[t])
          robot.input ((0,0))
          robot.move  (0, 0)
          viewer.show(world,[[0,0]],M,Xx_temp,Xy_temp)
          
          
          #正規化処理
          yudo_sum = fsum(yudo)
          yudo,yudo_sum = yudoup(yudo,yudo_sum)     ####とても小さな浮動小数値をある程度まで大きくなるまで桁をあげる
          ###0ワリ対処処理
          yudo_max = max(yudo)  #最大尤度のパーティクルを探す
          yudo_summax = float(yudo_sum) / yudo_max
          for j in xrange(M):
            yudo[j] = float(float(yudo[j])/yudo_max) / yudo_summax
          
          #リサンプリング処理(一点のみ)
          ###確率サイコロ
          rand_c = random.random()        # Random float x, 0.0 <= x < 1.0
          #print rand_c
          pc_num = 0.0
          for i in xrange(M) : 
            pc_num = pc_num + yudo[i]
            if pc_num >= rand_c : 
              print t,int(Xt[t][0]),int(Xt[t][1]),int(degrees(XDt[t]))  #変更反映前のXtの確認用
              Xt[t] = (Xx_temp[i],Xy_temp[i])  #タプルの要素ごとに代入はできないため、タプルとして一気に代入
              XDt[t] = Xd_temp[i]
              rand_c = 1.1
          
          print t,int(Xt[t][0]),int(Xt[t][1]),int(degrees(XDt[t]))
          print t,int(Xt_true[t][0]),int(Xt_true[t][1]),degrees(XDt_true[t]),degrees(XDt_true[t])-360
          
          t = t-1
          
          #if t == -1:  ##動作確認用の無限ループ
          #    t = EndStep-1
        ########## ↑ ##### xt(教示時刻で場合分け)のサンプリング ##### ↑ ##########
        """
        
        """
        loop = 0
        if loop == 1:
          #サンプリングごとに各パラメータ値を出力
          fp = open('./data/' + filename + '/' + filename +'_samp'+ repr(iter)+'.csv', 'w')
          fp.write('sampling_data,'+repr(iter)+'\n')  #num_iter = 10  #イテレーション回数
          fp.write('Ct\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(Ct[i])+',')
          fp.write('\n')
          fp.write('It\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(It[i])+',')
          fp.write('\n')
          fp.write('Position distribution\n')
          for k in xrange(K):
            fp.write('Myu'+repr(k)+','+repr(Myu[k][0])+','+repr(Myu[k][1])+'\n')
          for k in xrange(K):
            fp.write('Sig'+repr(k)+'\n')
            fp.write(repr(S[k])+'\n')
          for c in xrange(L):
            fp.write('W'+repr(c)+','+repr(W[c])+'\n')
          for c in xrange(L):
            fp.write('phi_l'+repr(c)+','+repr(phi_l[c])+'\n')
          fp.write('pi'+','+repr(pi)+'\n')
          fp.close()
          fp_x = open('./data/' + filename +'/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
          for t in xrange(EndStep) : 
            fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
          fp_x.close()
        """
      
      
  ######################################################################
  ####                   ↑場所概念学習フェーズ↑                   ####
  ######################################################################
      
      
      loop = 1
      ########  ↓ファイル出力フェーズ↓  ########
      if loop == 1:
        print "--------------------"
        #最終学習結果を出力
        print u"\n- <COMPLETED> Learning of Location Concepts ver. NEW MODEL. -"
        print 'Sample: ' + str(sample)
        print 'Ct: ' + str(Ct)
        print 'It: ' + str(It)
        for c in xrange(L):
          print "W%d: %s" % (c,W[c])
        for k in xrange(K):
          print "myu%d: %s" % (k, str(Myu[k].T))
        for k in xrange(K):
          print "sig%d: \n%s" % (k, str(S[k]))
        print 'pi: ' + str(pi)
        for c in xrange(L):
          print 'phi' + str(c) + ':',
          print str(phi_l[c])
        
        print "--------------------"
        
        #サンプリングごとに各パラメータ値を出力
        if loop == 1:
          fp = open('./data/' + filename +'/' + filename +'_kekka_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
          fp.write('sampling_data,'+repr(iter+1)+'\n')  #num_iter = 10  #イテレーション回数
          fp.write('Ct\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(Ct[i])+',')
          fp.write('\n')
          fp.write('It\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(It[i])+',')
          fp.write('\n')
          fp.write('Position distribution\n')
          for k in xrange(K):
            fp.write('Myu'+repr(k)+','+repr(Myu[k][0][0])+','+repr(Myu[k][1][0])+'\n')
          for k in xrange(K):
            fp.write('Sig'+repr(k)+'\n')
            fp.write(repr(S[k])+'\n')
          
          for c in xrange(L):
            fp.write(',')
            for i in xrange(len(W_index)):
              fp.write(W_index[i] + ',')   #####空白が入っているものがあるので注意(', ')
            fp.write('\n')
            fp.write('W'+repr(c)+',')
            for i in xrange(len(W_index)):
              fp.write(repr(W[c][i])+',')
            fp.write('\n')
          for c in xrange(L):
            fp.write(',')
            for k in xrange(K):
              fp.write(repr(k)+',')
            fp.write('\n')
            fp.write('phi_l'+repr(c)+',')
            for k in xrange(K):
              fp.write(repr(phi_l[c][k])+',')
            fp.write('\n')
          fp.write(',')
          for c in xrange(L):
            fp.write(repr(c)+',')
          fp.write('\n')
          fp.write('pi'+',')
          for c in xrange(L):
            fp.write(repr(pi[c])+',')
          fp.write('\n')
          fp.close()
          #fp_x = open('./data/' + filename +'/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
          #for t in xrange(EndStep) : 
          #  fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
          #fp_x.close()
        
        
        
        
        #各パラメータ値、初期値を出力
        fp_init = open('./data/' + filename +'/' + filename + '_init_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp_init.write('init_data\n')  #num_iter = 10  #イテレーション回数
        fp_init.write('L,'+repr(L)+'\n')
        fp_init.write('K,'+repr(K)+'\n')
        fp_init.write('alpha,'+repr(alpha)+'\n')
        fp_init.write('gamma,'+repr(gamma)+'\n')
        fp_init.write('bata0,'+repr(beta0)+'\n')
        fp_init.write('kappa0,'+repr(kappa0)+'\n')
        fp_init.write('m0,'+repr(m0)+'\n')
        fp_init.write('V0,'+repr(V0)+'\n')
        fp_init.write('nu0,'+repr(nu0)+'\n')
        fp_init.write('sigma_init,'+repr(sig_init)+'\n')
        fp_init.write('M,'+repr(M)+'\n')
        fp_init.write('N,'+repr(N)+'\n')
        fp_init.write('TN,'+repr(TN)+'\n')
        fp_init.write('Ct_init\n')
        for i in xrange(N):
          fp_init.write(repr(i)+',')
        fp_init.write('\n')
        for i in xrange(N):
          fp_init.write(repr(Ct_init[i])+',')
        fp_init.write('\n')
        fp_init.write('It_init\n')
        for i in xrange(N):
          fp_init.write(repr(i)+',')
        fp_init.write('\n')
        for i in xrange(N):
          fp_init.write(repr(It_init[i])+',')
        fp_init.write('\n')
        fp_init.write('Position distribution_init\n')
        for k in xrange(K):
          fp_init.write('Myu_init'+repr(k)+','+repr(Myu_init[k][0])+','+repr(Myu_init[k][1])+'\n')
        for k in xrange(K):
          fp_init.write('Sig_init'+repr(k)+'\n')
          fp_init.write(repr(S_init[k])+'\n')
        for c in xrange(L):
          fp_init.write('W_init'+repr(c)+','+repr(W_init[c])+'\n')
        #for c in xrange(L):
        #  fp_init.write('phi_l_init'+repr(c)+','+repr(phi_l_init[c])+'\n')
        #fp_init.write('pi_init'+','+repr(pi_init)+'\n')
        for c in xrange(L):
          fp_init.write(',')
          for k in xrange(K):
            fp_init.write(repr(k)+',')
          fp_init.write('\n')
          fp_init.write('phi_l_init'+repr(c)+',')
          for k in xrange(K):
            fp_init.write(repr(phi_l_init[c][k])+',')
          fp_init.write('\n')
        fp_init.write(',')
        for c in xrange(L):
          fp_init.write(repr(c)+',')
        fp_init.write('\n')
        fp_init.write('pi_init'+',')
        for c in xrange(L):
          fp_init.write(repr(pi_init[c])+',')
        fp_init.write('\n')
        
        fp_init.close()
        
        ##自己位置推定結果をファイルへ出力
        #filename_xt = raw_input("Xt:filename?(.csv) >")  #ファイル名を個別に指定する場合
        #filename_xt = filename
        #fp = open('./data/' + filename +'/' + filename_xt + '_xt_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        #fp2 = open('./data/' + filename_xt + '_xt_true.csv', 'w')
        #fp3 = open('./data/' + filename_xt + '_xt_heikatsu.csv', 'w')
        #fp.write(Xt)
        #for t in xrange(EndStep) : 
        #    fp.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
        #    #fp2.write(repr(Xt_true[t][0]) + ', ' + repr(Xt_true[t][1]) + '\n')
        #    #fp2.write(repr(Xt_heikatsu[t][0]) + ', ' + repr(Xt_heikatsu[t][1]) + '\n')
        #fp.writelines(repr(Xt))
        #fp.close()
        #fp2.close()
        #fp3.close()
        
        ##認識発話単語集合をファイルへ出力
        #filename_ot = raw_input("Otb:filename?(.csv) >")  #ファイル名を個別に指定する場合
        filename_ot = filename
        fp = open('./data/' + filename +'/' + filename_ot + '_ot_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp2 = open('./data/' + filename +'/' + filename_ot + '_w_index_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for n in xrange(N) : 
            for j in xrange(len(Otb[n])):
                fp.write(Otb[n][j] + ',')
            fp.write('\n')
        for i in xrange(len(W_index)):
            fp2.write(repr(i) + ',')
        fp2.write('\n')
        for i in xrange(len(W_index)):
            fp2.write(W_index[i] + ',')
        fp.close()
        fp2.close()
        
        print 'File Output Successful!(filename:'+filename+ "_" +str(iteration) + "_" + str(sample) + ')\n'
      
      
      ##パラメータそれぞれをそれぞれのファイルとしてはく
      if loop == 1:
        fp = open('./data/' + filename +'/' + filename + '_Myu_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(float(Myu[k][0][0]))+','+repr(float(Myu[k][1][0])) + '\n')
        fp.close()
        fp = open('./data/' + filename +'/' + filename + '_S_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(S[k][0][0])+','+repr(S[k][0][1])+','+repr(S[k][1][0]) + ','+repr(S[k][1][1])+'\n')
        fp.close()
        fp = open('./data/' + filename +'/' + filename + '_W_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for i in xrange(len(W_index)):
            fp.write(repr(W[c][i])+',')
          fp.write('\n')
          #fp.write(repr(W[l][0])+','+repr(W[l][1])+'\n')
        fp.close()
        fp = open('./data/' + filename +'/' + filename + '_phi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for k in xrange(K):
            fp.write(repr(phi_l[c][k])+',')
          fp.write('\n')
        fp.close()
        fp = open('./data/' + filename +'/' + filename + '_pi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          fp.write(repr(pi[c])+',')
        fp.write('\n')
        fp.close()
        
        fp = open('./data/' + filename +'/' + filename + '_Ct_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(Ct[t])+',')
        fp.write('\n')
        fp.close()
        
        fp = open('./data/' + filename +'/' + filename + '_It_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(It[t])+',')
        fp.write('\n')
        fp.close()
      
      ########  ↑ファイル出力フェーズ↑  ########
      
      
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
      
      loop = 1 #メインループ用フラグ
      while loop:
        #MAINCLOCK.tick(FPS)
        events = pygame.event.get()
        for event in events:
            if event.type == KEYDOWN:
                if event.key  == K_ESCAPE: exit()
        viewer.show(world,iti,0,[filename],[filename2])
        loop = 0
      
      
      
      

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
    
    print "[",
    for i in xrange(len(W_index[sample])):
      print "\""+ str(i) + ":" + str(W_index[sample][i]) + "\",",  #unicode(W_index[sample][i], 'shift-jis').encode('utf-8')
    print "]"
    
    
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
    #Ct = [ int(random.uniform(0,L)) for n in xrange(N)]
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
    """
    for j in xrange(len(W_index[sample])):
      for i in xrange(len(W_in)):
        if (W_index[sample][j] in W_in == False):
          W_out = W_out + [W_index[sample][j]]
    """
    
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
  """
    ###推定された場所概念番号を調べる
    L_dd = [0 for c in xrange(L)]
    for t in xrange(len(Ct)):
      for c in xrange(L):
        if Ct[t] == c:
          L_dd[c] = 1
    ##print L_dd #ok
  """
  
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
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
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
  





if __name__ == '__main__':
    import sys
    import os.path
    from __init__ import *
    from JuliusLattice_gmm import *
    import time
    
    
    filename = sys.argv[1]
    print filename
    
    #出力ファイル名を要求
    #filename = raw_input("trialname?(folder) >")
    
    
    Makedir( "data/" + filename )
    #Makedir( "data/" + filename + "/lattice" )
    
    #p0 = os.popen( "PATH=$PATH:../../latticelm" )  #パスを通す-＞通らなかった
    
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
    
    ##ループ後処理
    
    #p0.close()
    


########################################

