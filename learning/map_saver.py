#coding:utf-8
#map saver for each m_count
#Akira Taniguchi 2017/02/04-
import sys
import os
import signal
import subprocess
import time
import rospy
from std_msgs.msg import String
#import map_store.srv as map_store_srvs
from __init__ import *

trialname = sys.argv[1]
#datasetNUM = sys.argv[2]

mapsave = "rosrun map_server map_saver -f "
clocktime = 0.0

#trialname = "test" #raw_input("trialname?(output_folder) >")
#datasetNUM = 0 #raw_input("dataset_number?(0:albert,1:MIT) >")
#datasetname = datasets[int(datasetNUM)]
#datasetPATH = datasetfolder + datasetname

m_count = 0  #m_countの数
m_temp = 0

def callback(message):
  global clocktime
  global m_count
  global m_temp
  #print clocktime,rospy.get_time()
  
  #m_countのindexは1から始まる
  while (os.path.exists( datafolder + trialname + "/particle/" + str(m_count+1) + ".csv" ) == True):
    m_count += 1
    print "m_count",m_count, "m_temp",m_temp
  
  if (m_temp != m_count):
    if (float(clocktime) != float(rospy.get_time())):
      clocktime = rospy.get_time()
      #rospy.loginfo("%s", message)
      #rospy.loginfo("%s", clocktime)
      MSC = mapsave + datafolder + trialname + "/map/map"+ str(m_count)
      p = subprocess.Popen(MSC, shell=True) #, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
      print "Subprocess run map_saver.",clocktime
      fp = open( datafolder + trialname + "/map/map"+ str(m_count) + "_clocktime.txt", 'w')
      fp.write(str(clocktime))
      fp.close()
      time.sleep(2.0)
      #rospy.sleep(3.0)
      p.terminate()
      p.kill()
      m_temp += 1
      
  
time.sleep(2.0)
rospy.init_node('map_savering')
sub = rospy.Subscriber('clock', String, callback)
#rosbagがpauseの間はずっと同じ時刻を受け取り続ける


rospy.spin()


"""
#!/usr/bin/env python
#
# License: BSD
#   https://raw.github.com/robotics-in-concert/rocon_demos/license/LICENSE
#
def process_save_map(req):
    map_path = os.path.expanduser(rospy.get_param('map_path'))
    filename = map_path+rospy.get_param('filename', req.map_name)
    map_topic = rospy.get_param('map_topic', '/map')
    tmp_name = filename + '_ori'
    tmp_output = subprocess.check_output(['rosrun','map_server','map_saver','-f',tmp_name, 'map:=%s'%map_topic])
    rospy.sleep(2.0)
    tmp_name = tmp_name + '.yaml'
    crop_output = subprocess.check_output(['rosrun','map_server','crop_map',tmp_name,filename])
    rospy.loginfo('Map Saved into %s'%str(filename))
    #return map_store_srvs.SaveMapResponse()

if __name__ == '__main__':
    rospy.init_node('map_saver_with_crop',anonymous=True)
    srv_saver = rospy.Service('save_map', map_store_srvs.SaveMap, process_save_map)
rospy.spin()
"""