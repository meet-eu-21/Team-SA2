# Team-SA2

Meet-EU Team SA2, Topic A : Prediction of TADs

Based on results of the benchmark done on TADs prediction tools (Dali and Blanchette, 2015), we designed a CNN in order to discriminate TADs border, given a (1,33,33) windows from HiC data. In the aim of reconstructing full TADs, we also try to predict the length associated with a given border.

# Visuals
![Position scoring on chromosome X 12.5-17.5 kb](/Visuals/pos_score_meeteu_X.png)

## Requirements (Python 3.8)
* numpy          == 1.19.5 (https://numpy.org/doc/stable/index.html)
* scipy          == 1.7.1  (https://scipy.org/)
* tensorflow-gpu == 2.6.1  (https://www.tensorflow.org/install)
* HiCtoolbox               (available in the repository)


# Installation
Download model.zip file, unzip it at the location of your choice.

# Usage
In the predict.py file, variable "HiCfilename" (line 10),  enter the path of your HIC map.
You may want to change the name of the output file, this can be done line 41

Note: in order to use a CPU instead of a GPU, please change the shape format order from (1,33,33) to (33,33,1) as CPU does not support channel first.

# Authors and acknowledgments
Maxime Christophe, Wiam Mansouri, Antoine Szatkownik



