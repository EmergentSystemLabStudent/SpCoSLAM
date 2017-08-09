# encoding: shift_jis
#!/usr/bin/env python

#多項分布の確率を読み込んでカラーマップで表示するプログラム

#読み込むデータを指定
#場所の名前の音節列を読み込む→音素列に変換（単語辞書登録プログラムを一部流用）
#Wを読み込む→カラーマップで描画（縦軸が名前、横軸がindexCt）
#Φを読み込む→カラーマップで描画（縦軸がindexIt、横軸がindexCt）

###場所概念Ctはデータが語った番号だけにする
#Ctの結果を読み込み

###位置分布itもデータが語った番号だけ表示にする
#itの結果を読み込み→多項分布の確率を正規化

#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.colors import LogNorm
from __init__ import *

##学習データファイルから単語の読み込み


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

def MI_binary(b,W,pi,c,L,K):  #Mutual information(二値版):word_index、W、π、Ct
    #相互情報量の計算
    POC = W[c][b] * pi[c] #Multinomial(W[c]).pmf(B) * pi[c]   #場所の名前の多項分布と場所概念の多項分布の積
    PO = sum([W[ct][b] * pi[ct] for ct in xrange(L)]) #Multinomial(W[ct]).pmf(B)
    PC = pi[c]
    POb = 1.0 - PO
    PCb = 1.0 - PC
    PObCb = PCb - PO + POC
    POCb = PO - POC
    PObC = PC - POC
    #print POC,PO,PC,POb,PCb,PObCb,POCb,PObC
    
    # 相互情報量の定義の各項を計算
    temp1 = POC * log(POC/(PO*PC), 2)
    temp2 = POCb * log(POCb/(PO*PCb), 2)
    temp3 = PObC * log(PObC/(POb*PC), 2)
    temp4 = PObCb * log(PObCb/(POb*PCb), 2)
    score = temp1 + temp2 + temp3 + temp4
    return score
    

#ファイル名を要求
trialname = raw_input("dataname?>")
step = 50
iteration = step #int(raw_input("iteration_num?[1~10]>"))
#sample = int(raw_input("saple_num?[0~5]>"))
filename = datafolder + trialname + "/" + str(step)  ##FullPath of learning trial folder

iteration = iteration - 1
#w_indexの2行目を読み込む
#w_indexの2行目を読み込む
W_index= []
W_index_p = []

i = 0
#重みファイルを読み込み
for line in open(datafolder + trialname + '/'+ str(step) + '/weights.csv', 'r'):   ##読み込む
        #itemList = line[:-1].split(',')
        if (i == 0):
          MAX_Samp = int(line)
        i += 1

#最大尤度のパーティクル番号を保存
particle = MAX_Samp

"""
i = 0
#テキストファイルを読み込み
for line in open('./data/' + filename + '_w_index.csv', 'r'):   ##*_samp.100を順番に読み込む
    itemList = line[:-1].split(',')
    
    if(i == 1):
        for j in range(len(itemList)):
          W_index = W_index + [itemList[j]]
        
    i = i + 1
print W_index


LIST = []
hatsuon = [ "" for i in xrange(len(W_index)) ]
TANGO = []
##単語辞書の読み込み
for line in open("./web.000.htkdic", 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
      



##W_indexの単語を順番に処理していく
for c in xrange(len(W_index)):   
          W_index_sj = unicode(W_index[c], encoding='shift_jis')
          #if len(W_index_sj) != 1:  ##１文字は除外
          for moji in xrange(len(W_index_sj)):
              #print len(W_index_sj),str(W_index[i]),W_index_sj[moji]#,len(unicode(W_index[i], encoding='shift_jis'))
              for j in xrange(len(TANGO)):
                #print TANGO[j],j
                ##文字が最後でないとき and 次の文字が小文字のとき
                if (len(W_index_sj)-1 != moji) and (W_index_sj[moji+1] == u'ぁ' or W_index_sj[moji+1] == u'ぃ' or W_index_sj[moji+1] == u'ぅ' or W_index_sj[moji+1] == u'ぇ' or W_index_sj[moji+1] == u'ぉ' or W_index_sj[moji+1] == u'ゃ' or W_index_sj[moji+1] == u'ゅ' or W_index_sj[moji+1] == u'ょ'):   ##次の文字をみる
                    #print moji,W_index_sj[moji+1]
                    if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+W_index_sj[moji+1]):
                      print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]# + " "
                #文字が小文字のとき（そもそも存在しないため実行されない：消してもよい仕様）
                elif(W_index_sj[moji] == u'ぁ' or W_index_sj[moji] == u'ぃ' or W_index_sj[moji] == u'ぅ' or W_index_sj[moji] == u'ぇ' or W_index_sj[moji] == u'ぉ' or W_index_sj[moji] == u'ゃ' or W_index_sj[moji] == u'ゅ' or W_index_sj[moji] == u'ょ'):
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                    print W_index[c][moji] + " (komoji)"
                else:  ##文字が小文字でないとき and (文字が最後のとき or 次の文字が小文字でないとき)
                  #print moji , len(W_index_sj)-1,i
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                      print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
          print hatsuon[c]
          W_index_p = W_index_p + [hatsuon[c]]
          #else:
          #  print W_index[c] + " (one name)"
        
"""

##各場所の名前の単語ごとに
#meishi = meishi.encode('shift-jis')


#Ct = []
##Ctの読み込み
Ct,It = ReaditCtData(trialname, step, particle)

#Ct = []
###Ctの読み込み
#for line in open('./data/' + filename + '_Ct.csv', 'r'):
#   itemList = line[:-1].split(',')
#   
#   for i in xrange(len(itemList)):
#      if itemList[i] != '':
#        if Ct.count(int(itemList[i])) == 0:
#          Ct = Ct + [int(itemList[i])]

#It = []
##Ctの読み込み
#for line in open('./data/' + filename + '_It.csv', 'r'):
#   itemList = line[:-1].split(',')
#   
#   for i in xrange(len(itemList)):
#      if itemList[i] != '':
#        if It.count(int(itemList[i])) == 0:
#          It = It + [int(itemList[i])]



#######
N = max(It)#50#len(W_index_p)
L = max(Ct)#-1
K = max(It)#-1

pi = [ 0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
##phiの読み込み
phi = np.array([[0.0 for j in range(L)] for i in range(K)])
c = 0
it = 0
j = 0
#テキストファイルを読み込み
for line in open(datafolder + trialname + '/'+ str(step) + '/phi' + str(particle) + '.csv', 'r'):
        #f Ct.count(j) >= 1:
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        it = 0
        ip = 0
        for i in xrange(len(itemList)):
              ## if It.count(it) >= 1:
              if itemList[i] != '':
                #print c,i,itemList[i]
                phi[ip][c] = float(itemList[i])
                #print W[i][c]
                #print itemList
                ip = ip + 1
              ##it = it + 1
        c = c + 1
        #j = j + 1

"""
for c in range(L):
  phi_sum = 0.0
  for i in range(K):
    phi_sum = phi_sum + phi[i][c]
  phi_re = float( (1.0 - phi_sum) / len(It))
  for i in range(K):
    phi[i][c] = phi[i][c] + phi_re
"""
##piの読み込み

for line in open(datafolder + trialname + '/'+ str(step) + '/pi' + str(particle) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            pi[i] = float(itemList[i])

pi_temp = [pi[i] for i in range(len(pi))]
phi_l_temp = [ [phi[i][c] for c in xrange(L)] for i in xrange(K) ]
     
ni = [0 for i in range(len(pi))]
      
for i in range(len(pi)):
        ni[i] = pi[i]*((step)+L*alpha0)-alpha0

#phiのかさまし対策処理
phi = [ [( (phi[i][c]*(ni[c]+K*gamma0)-gamma0)+(gamma0/float(K)) ) / float( ni[c]+gamma0 ) for c in xrange(L)] for i in xrange(K) ]




plt.subplots_adjust(left=0.60, bottom=0.75, right=0.85, top=0.90, wspace=None, hspace=None)


#print C
plt.xlim([0,L]);
plt.ylim([N,0]);

ind = np.arange(L) # the x locations for the groups 
yyy = np.arange(N)
width = 0.50

Ct.sort()
It.sort()
plt.xticks(ind+width,([i for i in range(L)]), fontsize=8)#
plt.yticks(yyy+width, ([i+1 for i in range(K)]),fontsize=8)#, fontname='serif'
MAX = max([max(phi[i]) for i in range(len(phi))])
MIN = min([min(phi[i]) for i in range(len(phi))])
c = plt.pcolor(phi,cmap=plt.cm.hot,vmin=MIN,vmax=MAX) #gray
#title('')
#xlabel('Location Concepts ID', fontsize=24)
plt.xlabel('$C_t$', fontsize=10)
plt.ylabel('$i_t=k$', fontsize=10)


cbar=plt.colorbar(pad=0.08,shrink=0.95,aspect=12,ticks=[MIN,MAX]) #
#cbar.set_label('Frequency of estimated index number',fontsize=32)shrink=0.5,aspect=10,, orientation='horizontal'
cbar.ax.tick_params(labelsize=6)

plt.savefig(datafolder + trialname + '/'+'gs2_phi_'+ str(step) + '_0.eps', dpi=300, transparent=True)
plt.savefig(datafolder + trialname + '/'+'gs2_phi_'+ str(step) + '_0.png', dpi=300, transparent=True)
plt.show()
