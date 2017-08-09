#coding:utf-8
#rosbag play for learning and theaching
#スペースキー情報を送ることで一時停止と再開を行う。
#Akira Taniguchi 2017/02/03-
import sys
import os
import shutil
import signal
import subprocess
import time
import rospy
from std_msgs.msg import String
from __init__ import *

trialname = sys.argv[1]
datasetNUM = sys.argv[2]

#trialname = "test" #raw_input("trialname?(output_folder) >")
#datasetNUM = 0 #raw_input("dataset_number?(0:albert,1:MIT) >")
#datasetname = datasets[int(datasetNUM)]
#bagname = bags[int(datasetNUM)]
#datasetPATH = datasetfolder + datasetname + bagname

gflag = 1
#t_count = 0

#init.pyをコピー
shutil.copy("./__init__.py", datafolder + trialname )

#learn = "python ./learnSpCoSLAM2.py " + trialname + " " + str(datasetNUM)
#p = os.popen( learn )
#p.close()

SpCoSLAM = "python ./learnSpCoSLAM3.2NPYLM.py " + trialname + " " + str(datasetNUM)


def callback(message):
  global gflag
  flag = 0
  #if (gflag == 1):
  if (os.path.exists( datafolder + trialname + "/teachingflag.txt" ) == True):
    for line in open( datafolder + trialname + "/teachingflag.txt", 'r'):
      #itemList = line[:].split(',')
      flag = int(line)
      
  #print flag,gflag
  if (gflag == 1) and (flag == 1):
      print "Subprocess SpCoSLAM."
      p = subprocess.Popen(SpCoSLAM, shell=True)#, stdin=subprocess.PIPE) #, stdout=subprocess.PIPE)
      time.sleep(2.0)
      gflag = 0
  elif (gflag == 0) and (flag == 0):
      gflag = 1
  time.sleep(1.0)
  
  #  print len(teachingtime),t_count,flag
  #elif 
  #else:
  #  print "error."flag,t_count,ctime
  

time.sleep(2.0)
#p.stdin.write('%s\n' % " ")
#print "start." #,float(rospy.get_time()) #rosbagがstart直後のため時刻情報が取れない？
rospy.init_node('SpCoSLAM')
sub = rospy.Subscriber('clock', String, callback)
#if (flag == 1):  #rosbagを再開
#    flag = 0
#    p.stdin.write('%s\n' % " ")
#    print "start.",ctime
#    time.sleep(1.0)

rospy.spin()

#time.sleep(10.0)
#p.stdin.write('%s\n' % " ")
#print "pause."

#time.sleep(5.0)
#p.stdin.write('%s\n' % " ")
#print "start."

"""
if (len(teachingtime) == t_count):
  #よくわからないがどうやっても子プロセスが止められない。（Topicがpublishされたままになる）
  time.sleep(2.0)
  p.send_signal(signal.SIGINT)
  p.stdin.write("\n")
  
  #p = os.popen( learn )
  p.stdin.close()
  #p.stdout.close()
  
  # Get the process id
  pid = p.pid
  os.kill(pid, signal.SIGINT)
  
  p.terminate()
  p.kill()
  print "Done."
"""
