# Team-SA2

Meet-EU Team SA2

Topic A : Prediction of TADs


# Visuals
![Position scoring on chromosome X 12.5-17.5 kb](/Visuals/pos_score_meeteu_X.png)

# Context
MEET-EU 2021, SU TAD Group 2

# Installation

## Requirements (Python 3.8)
* h5py           == 3.1.0  (https://docs.h5py.org/en/stable/#)
* keras          == 2.6.0  (https://keras.io/)
* matplotlib     == 3.4.3  (https://matplotlib.org/)
* numpy          == 1.19.5 (https://numpy.org/doc/stable/index.html)
* pandas         == 1.3.4  (https://pandas.pydata.org/docs/index.html)
* scipy          == 1.7.1  (https://scipy.org/)
* tensorflow-gpu == 2.6.1  (https://www.tensorflow.org/install)
* cooler         == 0.8    (https://github.com/open2c/cooler)
* chromosight    == 1.6.1  (https://github.com/koszullab/chromosight)


# Usage
## Preprocess
As generating the input windows through the data generator is highly time consuming, it is highly recommended to pregenerate the data. In order to do so, the "Data_preprocess.ipynb" jupyter notebook will guide you (Feel free to modify the data generator if you don't want to locally store the input data). At the moment, the input windows size is (33,33). This parameter can be change while generating the data but cannot be dynamically passed (feature to be developped).
## Train CNN

The training is done with "run_cnn.py" file.
## Make prediction

## Evaluation

The evaluation of our results was made with Chromosight tool (original article : Matthey-Doret, C., Baudry, L., Breuer, A. et al. Computer vision for pattern detection in chromosome contact maps. Nat Commun 11, 5795 (2020). https://doi.org/10.1038/s41467-020-19562-7) 

A  raw2cool module has been designed in order to create the cool format expected by chromosight.

# Support

# Authors and acknowledgments
Maxime Christophe, Wiam Mansouri, Antoine Szatkownik

# License


