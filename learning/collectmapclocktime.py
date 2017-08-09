#c-od-i-n-:-ut-f---8
#!/usr/bin/env python

import sys
import string
#from sklearn.metrics.cluster import adjusted_rand_score
#import matplotlib.pyplot as plt
import numpy as np
#import math
from __init__ import *

trialname = raw_input("data_name? > ")

step = 50
m_count = 371

fp = open(datafolder + trialname + '/map/map_clocktime_all.csv', 'w')

for m in range(m_count):
  i = 0
  for line in open(datafolder + trialname + '/map/map' + str(m+1) + '_clocktime.txt', 'r'):
    #itemList = line[:-1].split(',')
    if (i == 0):
      fp.write(line)
      fp.write("\n")
    i = i + 1


print "close."
  
fp.close()
