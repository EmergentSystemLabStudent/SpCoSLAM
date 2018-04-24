Learning programs

【Folder】  
/lamg_m/: A folder including word dictionary files (Japanese syllable dictionary)


【Files】  
CNN_place.py： Save image features extracting by CNN (ver. Places-CNN)  
Julius1best_gmm.py： Using for evaluation　of place recognition rate (PRR). The speech files are recognized by the speech recognition Julius. It can get n-best speech recognition results.  
JuliusLattice_gmm.py： The speech files are recognized by the speech recognition Julius. It can get WFST speech recognition results for learning programs. 
README.txt： This file　　
SpCoSLAM.sh： A main execution script for online learning of SpCoSLAM　　
__init__.py：　A file for setting file paths and initial hyper-parameters  
autovisualization.py： A program for automatically drawing learning results sequentially
(Save can be done with screenshots etc.)
collectmapclocktime.py：　The times on the generated map files in one file collectively. For creating a movie.
gmapping.sh： Shell script for FastSLAM
learnSpCoSLAM3.2.py： SpCoSLAM online learning program (By setting of `__init__.py`, language model update and image features can be removed from SpCoSLAM)
map_saver.py： Saving a environmental map (using rospy)
new_place_draw.py： Visualization of position distributions (Gaussian distributions) on rviz 
new_place_draw_online.py： For online visualization of learning result
new_position_draw_online.py： For visualization of the robot position
run_SpCoSLAM.py： sub-program for performing SpCoSLAM
run_gmapping.sh： sub-program for performing gmapping
run_mapviewer.py： sub-program for performing map_server command (not used?)
run_mapviewer.sh： Shell script for performing run_mapviewer.py (not used?)
run_rosbag.py： sub-program for performing rosbag
run_roscore.sh： sub-program for performing roscore
saveSpCoMAP.rviz： rviz setting file
saveSpCoMAP_online.rviz： rviz setting file for online visualization


-----
[How to visualize the position distributions on rviz]
`roscore`
`rviz -d ./*/SpCoSLAM/learning/saveSpCoMAP_online.rviz `
`python ./autovisualization.py p30a20g10sfix008`

In case of individual specification
`rosrun map_server map_server ./p30a20g10sfix008/map/map361.yaml`
`python ./new_place_draw.py p30a20g10sfix008 50 23 `

-------------------------------------------------
更新日時
2017/02/12 Akira Taniguchi
2017/03/12 Akira Taniguchi
2018/01/12 Akira Taniguchi
2018/04/24 Akira Taniguchi
