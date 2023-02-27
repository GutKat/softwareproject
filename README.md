## Software project 2022/23:

This repository were created in the process of the course "053531 PR Softwareentwicklungsprojekt Bioinformatik (2022W)" in the winter-semester 2022/2023 at the university vienna. The project was supervised by Mag. Stefan Badelt and and Univ.-Prof. Dipl.-Phys. Dr. Ivo Hofacker.
The project focused on testing the deep-learning based prediction tool [UFold](https://github.com/uci-cbcl/UFold) in an unbiased way. For this purpose it also contains sequence designing to create artificial data, where we can control the bias within the data. 
UFold is is used for predicting secondary structure of RNAs. Some files within UFold were slightly changed and expanded, but most were left unchanged. Additional files were created to create data and test the performance of models.

## Prerequisites
--python >= 3.6.6

--torch >= 1.4 with cudnn >=10.0

--[munch](https://pypi.org/project/munch/2.0.2/)

--[subprocess](https://docs.python.org/3/library/subprocess.html)

--[collections](https://docs.python.org/2.7/library/collections.html#)

--matplotlib

--ViennRNA

--tqdm

--numpy

--pandas

--infrared

## Setup

The UFold folder contains the deep learning software [UFold](https://github.com/uci-cbcl/UFold). The file predict.py must be located within this folder. The file ml_forensic.py should be located within UFold/ufold/.

## Usage

#create_file.py


#predict.py
script to predict structures from randomly created sequence and store predictions in numpy files
created files are used for the script ml_forensic.py


#ml_forensic.py


#sequence_design.py

#sequence_design.ipynb


## 

## License

[MIT](https://choosealicense.com/licenses/mit/)




