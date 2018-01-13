This dataset was recorded 
 - by Cyrill Stachniss 
 - at the building 079 at the University of Freiburg, Germany
 - 22.09.2005 
 - iRobot B21r robot "Albert".


*.log file:

  Carmen-logfile (see logfile for format information) with 
   - IMAGE information
   - FLASER contains corrected pose in x, y, theta

*.png files:

  Image showing the information integrated from the SICK PLS laser
  range finder.


*.jpg files: 

  The images taken from the camera with a frame rate of 2-3Hz. The
  opening angle of the camera is around 65 degrees.
  

*.key files:

  Files containing the SIFT keys of the corresponding images. The file
  format starts with 2 integers giving the total number of keypoints
  and the size of descriptor vector for each keypoint (currently
  assumed to be 128). Then each keypoint is specified by 4 floating
  point numbers giving subpixel row and column location, scale, and
  orientation (in radians from -PI to PI).  Then the descriptor vector
  for each keypoint is given as a list of integers in range [0,255].


------------------------------------------------------------------------

./CNN_Place205/ folder:
This folder was created by Akira Taniguchi.
The csv files were extracted by Caffe using a reference model Places-205.
These visual features come from albert-B-laser-vision-dataset.
