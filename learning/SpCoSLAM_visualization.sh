#! /bin/sh
#Akira Taniguchi 2017/02/28-


echo -n "trialname?(output_folder) >"
read trialname

datasetNUM=0

gnome-terminal --command './run_roscore.sh'
sleep 3
gnome-terminal --command 'rviz -d ./saveSpCoMAP_online.rviz'
sleep 3
gnome-terminal --command 'python ./autovisualization.py '$trialname'

#gnome-terminal --command 'python ./run_mapviewer.sh '$trialname' '$m_count
#gnome-terminal --command 'python ./run_draw.py '$trialname' '$step
