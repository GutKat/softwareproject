## Software project 2022/23:

This repository were created in the process of the course "053531 PR Softwareentwicklungsprojekt Bioinformatik (2022W)" in the winter-semester 2022/2023 at the university vienna. The project was supervised by Mag. Stefan Badelt and Univ.-Prof. Dipl.-Phys. Dr. Ivo Hofacker.
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

The UFold folder contains the deep learning software [UFold](https://github.com/uci-cbcl/UFold). The files predict.py and ml_forensic.py must be located within this folder.

## Usage

### create_file.py
This scripts contains the functions:
-- random_sequence
    creates one random sequence
    Example Usage:
      random_sequence(length=100)
    
-- generate_random_seq_and_ss
    creates random sequence and corresponding structure using ViennaRNA from list of lengths
    Example Usage:
      generate_random_seq_and_ss(lengths=[70, 100, 120])
    
-- sample2bpseq
    creates a bpseq file from given sequence and structure under given path
    Example Usage:
      sample2bpseq(seq='GCCGUCGCGU', ss='((....)).., path='example_bpseq.txt')
    
-- random_bpseq
    creates a folder with bpseq files of given number of random sequences with given length, purpose or seed can be specified as well as the output folder
    Example Usage:
      random_bpseq(N_seqs = 1000, n_seq = 100, purpose = "train")
    
-- random_ml_forensic
    creates random sequences with given length and saves them in numpy files (needed for analysis done in ml_forensic.py)
    Example Usage:
      random_ml_forensic(N_seqs = 1000, n_seq = 100, output_folder = "ml_forensic/N1000_n100", seed=42)
      
-- fa2npy
    converts a given fasta file to a numpy file
    Example Usage:
      fa2npy(fa_file = "example_fasta.fa", output_path = "example_numpy.npy")
      
-- pickle2fa
    converts a given pickle file to a fasta file
    Example Usage:
      fa2npy(pickle_file = "example_pickle.pickle", fa_file= "example_fasta.fa")
      
The functions can be executed within the python script.

predict.py
script to predict structures from randomly created sequence and store predictions in numpy files
Created files are used for the script ml_forensic.py
Parameters for specifying the model, which should be used for prediction, and path to the file, which should be used for the prediction, can be specified within the python script.

ml_forensic.py
contains the function for analysing:
-- length_analysis
    
-- model_eval
-- structural_elements
-- known_structure_test
The functions can be executed within the python script.

sequence_design.py
creates new sequences from input fasta file. Can be executed from the command line:
--input or -i (str): path to the input fasta file
--output or -o (str): path to the output txt file
--design or -d (str): can be 1, 2 or 3. defines the design approach which should be used, default = 1
design argument defines the sequence design approach which should be used. 
1 = simple sequence design
2 = frequency based sequence design
3 = constraint generation sequence design

sequence_design.ipynb
creates new sequences from input fasta file (similar to sequence_design.py). File was used to test the sequence design approaches, since the needed packages were incompatible with the used computer.
Can only be executed within jupyter notebook script.


## 

## License

[MIT](https://choosealicense.com/licenses/mit/)




