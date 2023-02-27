# -*- coding: utf-8 -*-
'''
Author: Katrin Gutenbrunner
These functions have been created and used in the purpose of the Softwareproject 2022/23
This scripts contains functions, which convert files into a specific file format (bpseq), which is used by UFold for
training, testing and predicting, or convert pickle files back to fasta files or create random data and store it in specific files (bbseq for UFold or npy for ml_fornsic)
'''

import random
import numpy as np
from ufold import utils
import os
from tqdm import tqdm
import collections
import re
import pandas as pd
from pathlib import Path
import RNA

alphabet = ["A", "C", "G", "U"]
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

def random_sequence(length):
    '''function creates random sequence with given length

    The functions takes in the length and creates a random RNA sequence of the given length,
    which is then returned

    Args:
        length (int): the length of the random sequence

    Returns:
        sequence (str): the random created sequence with given length
    '''
    sequence = ''.join(random.choice('ACGU') for _ in range(length))
    return sequence


def generate_random_seq_and_ss(lengths):
    '''function creates random sequences and secondary structure from list of lengths

    The functions takes in a list of lengths and creates random RNA sequences of the given lengths and the
    corresponding secondary structure which is then returned as a list.

    Args:
        lengths (list): list with lengths, which the new created sequences and structure should have

    Returns:
        sequences (list): list with the created sequences and their structure, saved as tuple inside the list
    '''
    sequences = []
    for length in lengths:
        seq = random_sequence(length)
        sequences.append((seq, *RNA.fold(seq)))
    return sequences


class random_sample():
    """class for random RNA sample

    Attributes:
        seq (str): RNA sequence of the sample, created randomly
        ss (str): secondary structure of the sample, created using RNAFold from ViennaRNA

    """
    def __init__(self, length):
        """creates random RNA sequence and the corresponding secondary structure

        Args:
            length (int): length of the random sequence
        """
        seq, ss, energy = generate_random_seq_and_ss([length])[0]
        self.seq = seq
        self.ss = ss



def sample2bpseq(seq, ss, path):
    '''function to save RNA sample in a bpseq file

    creates a base pair sequence (bpseq) file from given sequence and structure under the given path.
    bbseq files are needed for creating (c)Pickles files (which are used by Ufold)

    Args:
        seq (str): sequence of RNA
        ss (str): secondary structure of RNA
        path (str): path and name under which the bpseq file should be saved

    Returns:
        None: creates bpseq file
    '''
    #create the pairs of the secondary structure
    pairs = utils.ct2struct(ss)
    paired = [0] * len(ss)
    #create the pair for the bbseq file
    for pair in pairs:
        le = pair[0]
        ri = pair[1]
        paired[le] = ri + 1
        paired[ri] = le + 1
    with open(path, "w") as f:
        n = len(seq)
        for index in range(n):
            line = [str(index+1), str(seq[index]), str(paired[index])]
            line = " ".join(line)
            f.write(line)
            if index != n-1:
                f.write("\n")
    return None


def random_bpseq(N_seqs, n_seq, purpose="train", seed_set=False, folder_path=None):
    '''creates a folder with bpseq files of N_seqs random sequences of length n_seq

    function creates a folder with bpseq files of N_seqs random sequences of length n_seq.
    The purpose of the bpseq file can be set to one of the following: ["train", "test", "val"],
    where the corresponding purpose will set a seed, which should be used for reproducibility.
    One can also set a seed with seed_set, where a specific seed can be elected.
    Furthermore a path to the folder can be fixed as folder_path, where the bpseq files will be saved.
    If no path is given, a new folder will be created, where the name includes the set seed or the purpose
    (depending on what is given), the numbers of sequences and their lengths.


    Args:
        N_seqs (int): number of sequences
        n_seq (int): length of each sequence
        purpose (str, optional): sets specific seeds for reproducibility, accepts "val" (validation), "test" (testing), "train" (training); default = "train"
        seed_set (int, optional): instead of purpose also a specific seed can be set, used for reproducibility, default = "False"
        folder_path (str, optional): path to the folder in which the bbseq files should be stored, if no folder_path is given, a folder is create with the given purpose or seed. default = None

    Returns:
        folder_path (str): path to the folder, where the bpseq file are saved.
    '''
    #set seed according to the parameter, either purpose or seed_set
    seed_dict = {"val": 10, "test": 20, "train": 30}
    seed = seed_dict[purpose]
    if seed_set:
        seed = seed_set
    utils.seed_torch(seed)
    random.seed(seed)

    # check if specific folder is given if not, create folder according to purpose or seed
    if folder_path:
        # check if folder already exist, if not create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    else:
        if seed_set:
            folder_path = f"N{N_seqs}_n{n_seq}_{seed}"
        else:
            folder_path = f"N{N_seqs}_n{n_seq}_{purpose}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # create the N_seqs sequences
    for _ in tqdm(range(N_seqs)):
        #create the sample with length n_seq
        sample = random_sample(n_seq)
        sample2bpseq(sample.seq, sample.ss, f"{folder_path}/len{n_seq}_{_ + 1}.txt")
    print(f"Finished creating {folder_path}")
    return folder_path



def random_ml_forensic(N_seqs, n_seq, output_folder, seed=42):
    '''function to create files for ml_forensic.py

    function takes in the number of sequences to create (N_seqs), length of sequence (n_seq), the folder where the file should be saved (output_folder)
    and optional a seed (seed) for reproducibility. The function creates the following files and stores them in the output fodler:
        - txt files with sequences with name "{N_seqs}_{n_seq}.txt"
        - npy file with sequences with name "{N_seqs}_{n_seq}_sequence.npy"
        - npy file with secondary structures with name "{N_seqs}_{n_seq}_structure.npy"

    Args:
        N_seqs (int): number of sequences
        n_seq (int): length of each sequence
        output_folder (str): path to the folder in which the files should be stored
        seed (int, optional): sets seed for reproducibility, default = 42

    Returns:
        None
            creates the txt file with name "N_seqs_n_seq.txt"
            creates the npy file with name "N_seqs_n_seq_sequence.npy"
            creates the npy file with name "N_seqs_n_seq_structure.npy"
    '''
    #set the given seed
    utils.seed_torch(seed)
    random.seed(seed)

    #create the sampels
    data = []
    for _ in tqdm(range(N_seqs)):
        data.append(random_sample(n_seq))

    #create the filename within the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    seq_txt_file = f"{output_folder}/N{N_seqs}_n{n_seq}.txt"
    sequence_file = output_folder + f"/N{N_seqs}_n{n_seq}_sequence.npy"
    structure_file = output_folder + f"/N{N_seqs}_n{n_seq}_structure.npy"

    seqs = []
    structures = []
    #write all sequences of the samples within txt_file
    with open(seq_txt_file, "w") as f:
        for i, dat in tqdm(enumerate(data)):
            f.write(f">random_{i+1}\n")
            f.write(f"{dat.seq}\n")#{dat.ss}
            f.write("\n")
            seqs.append(dat.seq)
            structures.append(dat.ss)

    #store sequences and structures within the sequence_file and structure file
    np.save(sequence_file, seqs)
    np.save(structure_file, structures)


def fa2npy(fa_file, output_path):
    '''converts fasta file to npy files

    function to create files for ml_forensic.py of an existing fasta file. Creates under the given output path (output_path=:
        - txt files with sequences
        - npy file with sequences
        - npy file with secondary structures

    Args:
        fa_file (str): path to the fasta file, which should be used
        output_path (str): path under which tge files should be stored

    Returns:
        None
            creates the txt file under "{output_path}.txt"
            creates the npy file under "{output_path}_sequence.npy"
            creates the npy file under "{output_path}_structure.npy"
    '''
    #load the fa file
    with open(fa_file, "r") as f:
        data = f.readlines()

    #pattern to get name of sequences
    pattern = ">(.*) en"

    #if folder path is not existing, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_stem = Path(fa_file).stem

    #create names for new files
    sequence_file = output_path + f"/{file_stem}_sequence.npy"
    structure_file = output_path + f"/{file_stem}_structure.npy"
    new_file = output_path + f"/{file_stem}.txt"

    #storing sequence and structures
    seqs = []
    structures = []

    #write into the new file
    with open(new_file, "w") as f:
        for i in tqdm(range(0, len(data), 3)):
            name = re.search(pattern, data[i])[1]
            seq = data[i + 1]
            seq = seq.replace("\n", "")
            ss = data[i + 2]
            ss = ss.replace("\n", "")
            seqs.append(seq)
            structures.append(ss)
            f.write(f">{name}\n")
            f.write(f"{seq}\n")  # {dat.ss}
            f.write("\n")
    #store sequences and structures in npy files
    np.save(sequence_file, seqs)
    np.save(structure_file, structures)


def pickle2fa(pickle_file, fa_file):
    '''converts pickle file to fasta file

    function takes in the path to the pickle file (pickle_file) and converts it into a fasta file, which is stored under the given path (fa_file)

    Args:
        pickle_file (str): path to the pickle file, from which the data should be collected
        fa_file (str): path to the fa file, in which the data should be stored

    Returns:
        None
            stores fasta file as "{fa_file}.fa"
    '''
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    #load data from pickle_file
    data = pd.read_pickle(pickle_file)
    names = []
    sequences = []
    structures = []
    for obj in tqdm(data):
        seq = obj[0]
        length = obj[2]
        name = obj[3]
        pairs = obj[4]
        #convert seq_encoding to sequences
        seq = utils.encoding2seq(seq)[0:length]
        #only use sequences with defined nucleotides (A,C,G or U)
        if "." in seq:
            continue

        #create the secodnary structure from the pair list
        ss = np.array(list(length * "."))
        for pair in pairs:
            if pair[0] < pair[1]:
                ss[pair[0]] = "("
                ss[pair[1]] = ")"
            else:
                ss[pair[1]] = "("
                ss[pair[0]] = ")"
        ss = "".join(ss)

        #save name sequences and structures
        names.append(name)
        sequences.append(seq)
        structures.append(ss)

    #if folder path is not existing, create it
    folder = str(Path(fa_file).parent)
    if not os.path.exists(folder):
        os.makedirs(folder)


    #create the fa file
    with open(fa_file, "w") as f:
        for i in range(len(names)):
            if i != len(names):
                f.write(f">{names[i]}\n")
                f.write(f"{sequences[i]}\n")
                f.write(f"{structures[i]}\n")
            # if we get to the last entry, we do not want to write a new line (\n)
            else:
                f.write(f">{names[i]}\n")
                f.write(f"{sequences[i]}\n")
                f.write(f"{structures[i]}")


if __name__ == '__main__':
    print(__doc__)
    pass
    # for n in [70, 100]:
    #     random_ml_forensic(2000, n, "data/analysis/type_analysis/ufold_model/", seed=42)