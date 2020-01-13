# SpCoSLAM

Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)  
This repository includes the source codes used for the experiments in our paper on IROS 2017.  
  
【Other repositories】  
 [NEW!] [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (Implementation for new version of online learning algorithm)   
 [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots in ROS   
 [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  

## Abstract of SpCoSLAM
We propose an online learning algorithm based on a Rao-Blackwellized particle filter for spatial concept acquisition and mapping. We have proposed a nonparametric Bayesian spatial concept acquisition model (SpCoA). We propose a novel method (SpCoSLAM) integrating SpCoA and FastSLAM in the theoretical framework of the Bayesian generative model. The proposed method can simultaneously learn place categories and lexicons while incrementally generating an environmental map.   

Figure: The graphical model of SpCoSLAM   
<img src="https://github.com/a-taniguchi/SpCoSLAM/blob/master/img/graphicalmodel02.jpg" width="520px">

## 【Execution environment】  
- Ubuntu 14.04  
- Python 2.7.6  
- ROS indigo  
- CNN feature extracter: Caffe (Reference model: [Places-205](http://places.csail.mit.edu/))  
- Speech recognition system: Julius dictation-kit-v4.3.1-linux (Using Japanese syllabary dictionary, lattice output)  
- If you perform the lexical acquisition (unsupervised word segmentaiton): [latticelm 0.4](http://www.phontron.com/latticelm/) and OpenFST  

In our paper of IROS2017, we used a rosbag file of open-dataset [albert-B-laser-vision-dataset](https://dspace.mit.edu/handle/1721.1/62291).

## 【Preparation for execution】  
- Path specification of training dataset, matching ros topic name etc (`__init__.py` and `run_gmapping.sh`)
- Create a file that stores the teaching time from the time information of the training dataset
- Prepare speech data files. Specify the file path in `__init__.py`  
- Start `CNN_place.py` before running the learning program  
  Create a folder for files of image features  
- To specify the number of particles, you need to change both `__ init__.py` and `run_gmapping.sh`  
- Change the path of the folder name in `/catkin_ws/src/openslam_gmapping/gridfastslam/gridslamprocessor.cpp`    
  We changed this file only.  
  [Note] If the original `gmapping` has already been installed on your PC, you need to change the uninstallation or path setting of `gmapping`.

## 【Execution procedure】
`cd ~/SpCoSLAM/learning `  
`./SpCoSLAM.sh `  
->trialname?(output_folder) >*output_folder_name* 

## 【Notes】
- Sometimes `gflag`-related errors sometimes appear in `run_rosbag.py`. 
  It is due to file reading failure. 
  It will reload and it will work so it will not be a problem.
- On low spec PCs, processing of gmapping can not catch up and maps can not be done well.

- This repository contains `gmapping`.
  The following files of `./catkin_ws/src/` folder follow the license of the original version of gmapping (License: CreativeCommons-by-nc-sa-2.0).

---
If you use this program to publish something, please describe the following citation information.

Reference:  
Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS2017), 2017.


Original paper:
https://arxiv.org/abs/1704.04664

Sample video:
https://youtu.be/z73iqwKL-Qk

2018/01/15  Akira Taniguchi  
2018/04/24  Akira Taniguchi (Update)  
2018/11/26  Akira Taniguchi (Update) 
