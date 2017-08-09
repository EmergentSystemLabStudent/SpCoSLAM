#! /bin/sh
#Akira Taniguchi 2017/02/03-2017/07/17


echo -n "trialname?(output_folder) >"
read trialname

#echo -n "dataset_number?(0:albert,1:MIT) >"
#read datasetNUM

#trialname=test3
datasetNUM=0

echo $trialname > /home/akira/Dropbox/SpCoSLAM/data/trialname.txt

mkdir /home/akira/Dropbox/SpCoSLAM/data/$trialname
mkdir /home/akira/Dropbox/SpCoSLAM/data/$trialname/particle
mkdir /home/akira/Dropbox/SpCoSLAM/data/$trialname/weight
mkdir /home/akira/Dropbox/SpCoSLAM/data/$trialname/map
mkdir /home/akira/Dropbox/SpCoSLAM/data/$trialname/img

SCAN=scan
gnome-terminal --command './run_roscore.sh'
sleep 5
gnome-terminal --command './run_gmapping.sh '$SCAN
gnome-terminal --command 'python ./map_saver.py '$trialname
gnome-terminal --command 'python ./run_rosbag.py '$trialname' '$datasetNUM
#cd ./learning
gnome-terminal --command 'python ./run_SpCoSLAM_LM_DNN.py '$trialname' '$datasetNUM

#python ./learning/learnSpCoSLAM2.py $trialname $datasetNUM


