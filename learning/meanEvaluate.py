#c-od-i-n-:-ut-f---8
#!/usr/bin/env python

import sys
import string
#from sklearn.metrics.cluster import adjusted_rand_score
#import matplotlib.pyplot as plt
import numpy as np
#import math
from __init__ import *

trialname = raw_input("data_name?(**_NUM) > ")
data_num1 = raw_input("data_start_num?(DATA_**) > ")
data_num2 = raw_input("data_end_num?(DATA_**) > ")
N = int(data_num2) - int(data_num1) +1
#filename = raw_input("Read_Ct_filename?(.csv) >")
S = int(data_num1)

step = 50
#N = step

NMIc_M = [[] for c in xrange(N)]
NMIi_M = [[] for c in xrange(N)]
PARs_M = [[] for c in xrange(N)]
PARw_M = [[] for c in xrange(N)]
L_M = [[] for c in xrange(N)]
K_M = [[] for c in xrange(N)]
ESEG_M = [[] for c in xrange(N)]
#K_M = [[] for c in xrange(N)]
#MM = [ np.array([[] for m in xrange(10) ]) for n in xrange(N)]
#MM_M = 

fp = open(datafolder + '/Evaluation/' + trialname + '_' + data_num1 + '_' + data_num2 + '_Evaluation.csv', 'w')
fp.write('NMIc,NMIi,PARs,PARw,L,K,ESEG\n')

i = 0
#NMIc_MAX = [[0,0]]
#NMIi_MAX = [[0,0]]
#PARs_MAX = [[0,0]]
#PARw_MAX = [[0,0]]
#PARs_MAX = [[0,0]]
#PARw_MAX = [[0,0]]

for s in range(N):
  i = 0
  for line in open(datafolder + trialname + str(s+1).zfill(3) + '/' + trialname + str(s+1).zfill(3) +'_meanEvaluation0MI.csv', 'r'):
    itemList = line[:-1].split(',')
    if (i != 0) and (itemList[0] != '') and (i <= step):
      #print i,itemList
      NMIc_M[s] = NMIc_M[s] + [float(itemList[2])]
      NMIi_M[s] = NMIi_M[s] + [float(itemList[3])]
      PARs_M[s] = PARs_M[s] + [float(itemList[4])]
      #PARw_M[s] = PARw_M[s] + [float(itemList[5])]
      L_M[s] = L_M[s] + [float(itemList[6])]
      K_M[s] = K_M[s] + [float(itemList[7])]
      ESEG_M[s] = ESEG_M[s] + [float(itemList[9])]
      #if (float(itemList[0]) > NMIc_MAX[0][1]):
      #    NMIc_MAX = [[s+1,float(itemList[0])]] + NMIc_MAX
      #if (float(itemList[1]) > NMIi_MAX[0][1]):
      #    NMIi_MAX = [[s+1,float(itemList[1])]] + NMIi_MAX
      #if (float(itemList[3]) > PARw_MAX[0][1]):
      #    PARw_MAX = [[s+1,float(itemList[3])]] + PARw_MAX
    i = i + 1
    
  i = 0
  for line in open(datafolder + trialname + str(s+1).zfill(3) + '/' + trialname + str(s+1).zfill(3) +'_meanEvaluationPARw.csv', 'r'):
    itemList = line[:-1].split(',')
    if (i != 0) and (itemList[4] != '') and (i <= step):
      #print i,itemList
      #NMIc_M[s] = NMIc_M[s] + [float(itemList[2])]
      #NMIi_M[s] = NMIi_M[s] + [float(itemList[3])]
      #PARs_M[s] = PARs_M[s] + [float(itemList[4])]
      PARw_M[s] = PARw_M[s] + [float(itemList[4])]
      #L_M[s] = L_M[s] + [float(itemList[6])]
      #K_M[s] = K_M[s] + [float(itemList[7])]
      #ESEG_M[s] = ESEG_M[s] + [float(itemList[9])]
      #if (float(itemList[0]) > NMIc_MAX[0][1]):
      #    NMIc_MAX = [[s+1,float(itemList[0])]] + NMIc_MAX
      #if (float(itemList[1]) > NMIi_MAX[0][1]):
      #    NMIi_MAX = [[s+1,float(itemList[1])]] + NMIi_MAX
      #if (float(itemList[3]) > PARw_MAX[0][1]):
      #    PARw_MAX = [[s+1,float(itemList[3])]] + PARw_MAX
    i = i + 1
    #print NMIc_M[s]
    #for i in xrange(len(itemList)):
    #   if itemList[i] != '':
         
    #MM[s] = MM[s] + [[float(itemList[0]),float(itemList[1]),float(itemList[2]),float(itemList[3])]]
    #NMIi = adjusted_rand_score(CtC, Ct)
    #print str(NMIi)
    #NMIi_M = NMIi_M + NMIi
 
  NMIc_M[s] = np.array(NMIc_M[s])
  NMIi_M[s] = np.array(NMIi_M[s])
  PARs_M[s] = np.array(PARs_M[s])
  PARw_M[s] = np.array(PARw_M[s])
  L_M[s] = np.array(L_M[s])
  K_M[s] = np.array(K_M[s])
  ESEG_M[s] = np.array(ESEG_M[s])
  #if (NMIc_M[s][-1] > NMIc_MAX[0][1]):
  #        NMIc_MAX = [[s+1,NMIc_M[s][-1]]] + NMIc_MAX
  #if (NMIi_M[s][-1] > NMIi_MAX[0][1]):
  #        NMIi_MAX = [[s+1,NMIi_M[s][-1]]] + NMIi_MAX
  #if (PARw_M[s][-1] > PARw_MAX[0][1]):
  #        PARw_MAX = [[s+1,PARw_M[s][-1]]] + PARw_MAX
  #if (PARs_M[s][-1] > PARs_MAX[0][1]):
  #        PARs_MAX = [[s+1,PARs_M[s][-1]]] + PARs_MAX
  print PARw_M[s],N,len(PARw_M),len(PARw_M[s])

#print "NMIc_MAX:",NMIc_MAX
#print "NMIi_MAX:",NMIi_MAX
#print "PARw_MAX:",PARw_MAX
#print "PARs_MAX:",PARs_MAX
#print NMIc_M
#MM_M = sum(MM)/N
NMIc_MM = sum(NMIc_M)/float(N)
NMIi_MM = sum(NMIi_M)/float(N)
PARw_MM = sum(PARw_M)/float(N)
PARs_MM = sum(PARs_M)/float(N)
L_MM = sum(L_M)/float(N)
K_MM = sum(K_M)/float(N)
ESEG_MM = sum(ESEG_M)/float(N)
#print NMIc_MM
#MI,NMIi,PARs,PARw,

for iteration in xrange(len(NMIc_MM)):
  fp.write( str(NMIc_MM[iteration])+','+ str(NMIi_MM[iteration])+','+ str(PARs_MM[iteration])+','+str(PARw_MM[iteration])+ ','+str(L_MM[iteration])+','+ str(K_MM[iteration])+','+str(ESEG_MM[iteration]) )
  fp.write('\n')
fp.write('\n')

#for iteration in xrange(10):
#  NMIc_MS = np.array([NMIc_M[s][iteration] for s in xrange(N)])
#  NMIc_std = np.std(NMIc_MS, ddof=1)
#  #print NMIc_std

for iteration in xrange(len(NMIc_MM)):
  NMIc_MS = np.array([NMIc_M[s][iteration] for s in xrange(N)])
  NMIi_MS = np.array([NMIi_M[s][iteration] for s in xrange(N)])
  PARs_MS = np.array([PARs_M[s][iteration] for s in xrange(N)])
  PARw_MS = np.array([PARw_M[s][iteration] for s in xrange(N)])
  L_MS = np.array([L_M[s][iteration] for s in xrange(N)])
  K_MS = np.array([K_M[s][iteration] for s in xrange(N)])
  ESEG_MS = np.array([ESEG_M[s][iteration] for s in xrange(N)])
  #print iteration,np.std(NMIc_MS, ddof=1)
  fp.write( str(np.std(NMIc_MS, ddof=1))+','+ str(np.std(NMIi_MS, ddof=1))+','+ str(np.std(PARs_MS, ddof=1))+','+str(np.std(PARw_MS, ddof=1)) +','+str(np.std(L_MS, ddof=1)) +','+str(np.std(K_MS, ddof=1)) +','+str(np.std(ESEG_MS, ddof=1)) )
  fp.write('\n')
#np.std
#float(NMIi_M / N)
#print "NMIi mean"
#print str(NMIi_M)
print "close."
  
fp.close()
