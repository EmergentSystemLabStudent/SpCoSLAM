# encoding: shift_jis
#!/usr/bin/env python

#�������z�̊m����ǂݍ���ŃJ���[�}�b�v�ŕ\������v���O����

#�ǂݍ��ރf�[�^���w��
#�ꏊ�̖��O�̉��ߗ��ǂݍ��ށ����f��ɕϊ��i�P�ꎫ���o�^�v���O�������ꕔ���p�j
#W��ǂݍ��ށ��J���[�}�b�v�ŕ`��i�c�������O�A������indexCt�j
#����ǂݍ��ށ��J���[�}�b�v�ŕ`��i�c����indexIt�A������indexCt�j

###�ꏊ�T�OCt�̓f�[�^��������ԍ������ɂ���
#Ct�̌��ʂ�ǂݍ���

###�ʒu���zit���f�[�^��������ԍ������\���ɂ���
#it�̌��ʂ�ǂݍ��݁��������z�̊m���𐳋K��

#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.colors import LogNorm
from __init__ import *

##�w�K�f�[�^�t�@�C������P��̓ǂݍ���


#it��Ct�̃f�[�^��ǂݍ��ށi�������������̂݁j
def ReaditCtData(trialname, cstep, particle):
  CT,IT = [0 for i in xrange(step)],[0 for i in xrange(step)]
  i = 0
  if (step != 0):  #�ŏ��̃X�e�b�v�ȊO
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

def MI_binary(b,W,pi,c,L,K):  #Mutual information(��l��):word_index�AW�A�΁ACt
    #���ݏ��ʂ̌v�Z
    POC = W[c][b] * pi[c] #Multinomial(W[c]).pmf(B) * pi[c]   #�ꏊ�̖��O�̑������z�Əꏊ�T�O�̑������z�̐�
    PO = sum([W[ct][b] * pi[ct] for ct in xrange(L)]) #Multinomial(W[ct]).pmf(B)
    PC = pi[c]
    POb = 1.0 - PO
    PCb = 1.0 - PC
    PObCb = PCb - PO + POC
    POCb = PO - POC
    PObC = PC - POC
    #print POC,PO,PC,POb,PCb,PObCb,POCb,PObC
    
    # ���ݏ��ʂ̒�`�̊e�����v�Z
    temp1 = POC * log(POC/(PO*PC), 2)
    temp2 = POCb * log(POCb/(PO*PCb), 2)
    temp3 = PObC * log(PObC/(POb*PC), 2)
    temp4 = PObCb * log(PObCb/(POb*PCb), 2)
    score = temp1 + temp2 + temp3 + temp4
    return score
    

#�t�@�C������v��
trialname = raw_input("dataname?>")
step = 50
iteration = step #int(raw_input("iteration_num?[1~10]>"))
#sample = int(raw_input("saple_num?[0~5]>"))
filename = datafolder + trialname + "/" + str(step)  ##FullPath of learning trial folder

iteration = iteration - 1
#w_index��2�s�ڂ�ǂݍ���
#w_index��2�s�ڂ�ǂݍ���
W_index= []
W_index_p = []

i = 0
#�d�݃t�@�C����ǂݍ���
for line in open(datafolder + trialname + '/'+ str(step) + '/weights.csv', 'r'):   ##�ǂݍ���
        #itemList = line[:-1].split(',')
        if (i == 0):
          MAX_Samp = int(line)
        i += 1

#�ő�ޓx�̃p�[�e�B�N���ԍ���ۑ�
particle = MAX_Samp

"""
i = 0
#�e�L�X�g�t�@�C����ǂݍ���
for line in open('./data/' + filename + '_w_index.csv', 'r'):   ##*_samp.100�����Ԃɓǂݍ���
    itemList = line[:-1].split(',')
    
    if(i == 1):
        for j in range(len(itemList)):
          W_index = W_index + [itemList[j]]
        
    i = i + 1
print W_index


LIST = []
hatsuon = [ "" for i in xrange(len(W_index)) ]
TANGO = []
##�P�ꎫ���̓ǂݍ���
for line in open("./web.000.htkdic", 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
      



##W_index�̒P������Ԃɏ������Ă���
for c in xrange(len(W_index)):   
          W_index_sj = unicode(W_index[c], encoding='shift_jis')
          #if len(W_index_sj) != 1:  ##�P�����͏��O
          for moji in xrange(len(W_index_sj)):
              #print len(W_index_sj),str(W_index[i]),W_index_sj[moji]#,len(unicode(W_index[i], encoding='shift_jis'))
              for j in xrange(len(TANGO)):
                #print TANGO[j],j
                ##�������Ō�łȂ��Ƃ� and ���̕������������̂Ƃ�
                if (len(W_index_sj)-1 != moji) and (W_index_sj[moji+1] == u'��' or W_index_sj[moji+1] == u'��' or W_index_sj[moji+1] == u'��' or W_index_sj[moji+1] == u'��' or W_index_sj[moji+1] == u'��' or W_index_sj[moji+1] == u'��' or W_index_sj[moji+1] == u'��' or W_index_sj[moji+1] == u'��'):   ##���̕������݂�
                    #print moji,W_index_sj[moji+1]
                    if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+W_index_sj[moji+1]):
                      print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]# + " "
                #�������������̂Ƃ��i�����������݂��Ȃ����ߎ��s����Ȃ��F�����Ă��悢�d�l�j
                elif(W_index_sj[moji] == u'��' or W_index_sj[moji] == u'��' or W_index_sj[moji] == u'��' or W_index_sj[moji] == u'��' or W_index_sj[moji] == u'��' or W_index_sj[moji] == u'��' or W_index_sj[moji] == u'��' or W_index_sj[moji] == u'��'):
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                    print W_index[c][moji] + " (komoji)"
                else:  ##�������������łȂ��Ƃ� and (�������Ō�̂Ƃ� or ���̕������������łȂ��Ƃ�)
                  #print moji , len(W_index_sj)-1,i
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                      print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
          print hatsuon[c]
          W_index_p = W_index_p + [hatsuon[c]]
          #else:
          #  print W_index[c] + " (one name)"
        
"""

##�e�ꏊ�̖��O�̒P�ꂲ�Ƃ�
#meishi = meishi.encode('shift-jis')


#Ct = []
##Ct�̓ǂݍ���
Ct,It = ReaditCtData(trialname, step, particle)

#Ct = []
###Ct�̓ǂݍ���
#for line in open('./data/' + filename + '_Ct.csv', 'r'):
#   itemList = line[:-1].split(',')
#   
#   for i in xrange(len(itemList)):
#      if itemList[i] != '':
#        if Ct.count(int(itemList[i])) == 0:
#          Ct = Ct + [int(itemList[i])]

#It = []
##Ct�̓ǂݍ���
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

pi = [ 0 for c in xrange(L)]     #�ꏊ�T�O��index�̑������z(L����)
##phi�̓ǂݍ���
phi = np.array([[0.0 for j in range(L)] for i in range(K)])
c = 0
it = 0
j = 0
#�e�L�X�g�t�@�C����ǂݍ���
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
##pi�̓ǂݍ���

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

#phi�̂����܂��΍􏈗�
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
