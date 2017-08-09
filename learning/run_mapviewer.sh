#! /bin/bash
#Akira Taniguchi 2017/02/28-

gnome-terminal --command 'python ./run_mapviewer.py '$1' '$2

#gnome-terminal --command 'rosrun map_server map_server /home/akira/Dropbox/SpCoSLAM/data/'$1'/map/map'$2'.yaml'
#rosrun map_server map_server /home/akira/Dropbox/SpCoSLAM/data/$1/map/map$2.yaml
