#coding:utf-8
#rosbag play for learning and theaching
#スペースキー情報を送ることで一時停止と再開を行う。
#Akira Taniguchi 2017/02/03-
import sys
import os
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
datasetname = datasets[int(datasetNUM)]
bagname = bags[int(datasetNUM)]
datasetPATH = datasetfolder + datasetname + bagname

flag = 0
t_count = 0
teachingtime = []
for line in open( datasetfolder + datasetname + 'teaching.csv', 'r'):
  #itemList = line[:].split(',')
  teachingtime.append(float(line))

fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
fp.write(str(flag))
fp.close()

fp = open( datafolder + trialname + "/gwaitflag.txt", 'w')
fp.write(str(0))
fp.close()

#learn = "python ./learnSpCoSLAM2.py " + trialname + " " + str(datasetNUM)
#p = os.popen( learn )
#p.close()

rosbag = "rosbag play -r " + str(rosbagSpeed) + " --clock "+ datasetPATH +" --pause"

p = subprocess.Popen(rosbag, shell=True, stdin=subprocess.PIPE) #, stdout=subprocess.PIPE)
print "Subprocess run rosbag."
#print p.communicate()
#, stderr=subprocess.PIPE)
#for line in p.stdout.readlines():
#    print line.strip()
#p.stdin.write(" ")


#教示時刻回数、一時停止処理
#for t in teachingtime:
def callback(endflag):
  global flag
  global t_count
  ctime = float(rospy.get_time())
  #int(ctime) == int(teachingtime[t_count]) and 
  if (len(teachingtime) != t_count):
    #print len(teachingtime),t_count,flag
    if (flag == 1): #and (int(endflag) == 1):  #rosbagを再開
      #flag = 0
      #fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
      #fp.write(str(flag))
      #fp.close()
      for line in open( datafolder + trialname + "/teachingflag.txt", 'r'):
        #itemList = line[:].split(',')
        gflag = int(line)
      if (gflag == 0):
        flag = 0
        p.stdin.write('%s\n' % " ")
        print t_count,"start!",ctime
        time.sleep(1.0)
    elif (int(ctime) == int(teachingtime[t_count]) and (flag == 0)):  #rosbagを一時停止
      #teachingtimeのフラグをファイル出力
      flag = 1
      fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
      fp.write(str(flag))
      fp.close()
      
      p.stdin.write('%s\n' % " ")
      print t_count,"pause.",ctime
      time.sleep(10.0)
      t_count += 1
  elif(len(teachingtime) == t_count): #最終
    if (flag == 1): #and (int(endflag) == 1):  #rosbagを再開
      #flag = 0
      #fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
      #fp.write(str(flag))
      #fp.close()
      for line in open( datafolder + trialname + "/teachingflag.txt", 'r'):
        #itemList = line[:].split(',')
        gflag = int(line)
      if (gflag == 0):
        flag = 0
        p.stdin.write('%s\n' % " ")
        print t_count,"start!",ctime
        time.sleep(1.0)
  #  print len(teachingtime),t_count,flag
  #elif 
  #else:
  #  print "error."flag,t_count,ctime
  

time.sleep(2.0)
p.stdin.write('%s\n' % " ")
print "start." #,float(rospy.get_time()) #rosbagがstart直後のため時刻情報が取れない？
rospy.init_node('play_rosbag')
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
