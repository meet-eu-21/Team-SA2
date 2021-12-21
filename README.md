# Team-SA2

Meet-EU Team SA2

Topic A : Prediction of TADs


# Visuals
![Position scoring on chromosome X 12.5-17.5 kb](/Visuals/pos_score_meeteu_X.png)

# Context
MEET-EU 2021, SU TAD Group 2

# Installation

## Packages (Python 3.8)
* h5py           == 3.1.0
* keras          == 2.6.0
* matplotlib     == 3.4.3
* numpy          == 1.19.5
* pandas         == 1.3.4
* scipy          == 1.7.1
* tensorflow-gpu == 2.6.1

# Usage
As generating the input windows through the data generator is highly time consuming, it is highly recommended to pregenerate the data. In order to do so, the "Data_preprocess.ipynb" jupyter notebook will guide you (Feel free to modify the data generator if you don't want to locally store the input data). At the moment, the input windows size is (33,33). This parameter can be change while generating the data but cannot be dynamically passed (feature to be developped).

The training is done with "run_cnn.py" file.

# Support

# Authors and acknowledgments
Maxime Christophe, Aïssata Kaba, Wiam Mansouri, Antoine Szatkownik

# License


