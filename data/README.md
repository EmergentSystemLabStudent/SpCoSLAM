# Learning results folder /data/  

- /x/: x is the number of learning steps (the number of teaching utterances)  
  - /fst_gmm/: WFST output folder of speech recognition results by Julius  
  - /out_gmm/:　Output folder for word segmentation results by latticelm  
  - index?.csv: the category number of spatial concepts and position distributions in ?-th particle (? is the number of particles)
  - mu?.csv: mean vectors of the position distribution 　
  - particle?.csv: particle information (the number of steps, particle ID, x coordinate of the robot, y coordinate of the robot, orientation of the robot, log likelihood, particle ID in previous step, index of spatial concept, index of position distribution)
  - phi?.csv: results of Multinomial distribution of index it of position distribution
  - pi?.csv: results of Multinomial distribution of index Ct of spatial concepts
  - sig?.csv: covariance matrix of the position distribution 
  - theta?.csv: Multinomial distribution of image feature
  - W_list?.csv: word list
  - W?.csv: Multinomial distribution of the names of places (The order follows the word list.)
  - WD.htkdic:　Learned word dictionary including initial Japanese syllables.
  - weights.csv: particle weights
- gwaitflag.txt: a flag argument for learning programs
- m_count2step.csv: relationships between m_count values and step values
- teachingflag.txt: a flag argument for learning programs
