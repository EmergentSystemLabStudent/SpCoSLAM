#!/usr/bin/env python
# -*- coding:utf-8 -*-
#Akira Taniguchi 2017/02/28-
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

trialname = sys.argv[1]
s =  sys.argv[2]

map = "rosrun map_server map_server /home/akira/Dropbox/SpCoSLAM/data/"+ trialname+"/map/map"+s+".yaml"

p = subprocess.Popen(map, shell=True, stdin=subprocess.PIPE)
#rosrun map_server map_server /home/akira/Dropbox/SpCoSLAM/data/$1/map/map$2.yaml
time.sleep(5.0)

#p.kill
#p.send_signal(signal.SIGINT)
#p.stdin.write(signal.SIGINT)