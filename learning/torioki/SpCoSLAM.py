#coding:utf-8
#Akira Taniguchi 2017/02/03-
import sys
import os
import signal
import subprocess
import time
from __init__ import *


trialname = "test" #raw_input("trialname?(output_folder) >")

datasetNUM = 0 #raw_input("dataset_number?(0:albert,1:MIT) >")
datasetname = datasets[int(datasetNUM)]
datasetPATH = datasetfolder + datasetname


#learn = "python ./learnSpCoSLAM2.py " + trialname + " " + str(datasetNUM)
#p = os.popen( learn )
#p.close()

rosbag = "rosbag play --clock ~/Dropbox/SpCoSLAM/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/albertBimgA.bag --pause"

p = subprocess.Popen(rosbag, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
print "Subprocess run rosbag."
#print p.communicate()
#, stderr=subprocess.PIPE)
#for line in p.stdout.readlines():
#    print line.strip()
time.sleep(5.0)
#p.stdin.write(" ")
p.stdin.write('%s\n' % " ")
print "start."
#for line in p.stdout.readlines():
#    print line.strip()
time.sleep(5.0)
p.stdin.write('%s\n' % " ")
print "pause."

#print p.communicate()
#for line in p.stdout.readlines():
#    print line.strip()

time.sleep(5.0)
p.stdin.write('%s\n' % " ")
print "start."
#time.sleep(5.0)
#p.stdin.write('%s\n' % " ")

#よくわからないがどうやっても子プロセスが止められない。（Topicがpublishされたままになる）
time.sleep(5.0)
p.send_signal(signal.SIGINT)
p.stdin.write("\n")

#p = os.popen( learn )

p.stdin.close()
p.stdout.close()

# Get the process id
pid = p.pid
os.kill(pid, signal.SIGINT)

p.terminate()
p.kill()
print "Done."

#time.sleep(10.0) #sleep(秒指定)

#echo -n "trialname?(output_folder) >"
#read bun

#mkdir ~/Dropbox/iCub/datadump/$bun
#mkdir ~/Dropbox/iCub/datadump/$bun/image





#gnome-terminal --tab --command 'yarp server --write' --tab --command 'iCub_SIM'

#sleep 10
#gnome-terminal --command './dumper.sh '$bun

#sleep 2
#./connect.sh $bun

