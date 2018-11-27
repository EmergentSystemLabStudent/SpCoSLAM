Please download "gmapping" package.

$ git clone https://github.com/ros-perception/slam_gmapping.git
$ git clone https://github.com/ros-perception/openslam_gmapping.git

Change "gridslamprocessor.cpp" file from the file of original gmapping to the file of github. 
catkin_ws/src/openslam_gmapping/gridfastslam/gridslamprocessor.cpp

Additions and changes in this cpp file shown as follows:
//s//Additions and changes/////////////////////////////////////////////
# Code of additions and changes
//e//Additions and changes/////////////////////////////////////////////


You also need to change the PATH in this cpp file.
---
Line 424: std::ifstream ifs(datafolder+"trialname.txt");
and
Line 432: string filename( datafolder );
---
datafolder is same to __init__.py

$ catkin_make
$ source devel/setup.bash
$ catkin_make install
$ rosmake gmapping
$ echo "source PATH/catkin_ws/devel/setup.bash" >> ~/.bashrc

PATH is a path of your PC.


[Caution] If original gmapping has already been installed, you need to change uninstall or path setting.

$ sudo apt-get purge ros-indigo-gmapping

