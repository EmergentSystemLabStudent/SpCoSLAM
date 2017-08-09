#! /bin/sh
#Akira Taniguchi 2017/02/03-
#botsu

rosbag play --clock ~/Dropbox/SpCoSLAM/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/albertBimgA.bag --pause
sleep 100
rosbag play --clock ~/Dropbox/SpCoSLAM/rosbag/albert-b-laser-vision/albert-B-laser-vision-dataset/albertBimgB.bag --pause
